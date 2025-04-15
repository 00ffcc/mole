from accelerate import Accelerator, ProfileKwargs
import torch.distributed as dist
import torch
# from mole.embedding_offload.embedding_adamw import SparseEmbedding
# from mole.embedding_offload.embedding_fix import SparseEmbedding
from mole.embedding_offload.embedding_fast import SparseEmbedding
from contextlib import nullcontext, contextmanager
from codetiming import Timer
from torch.utils.data import DataLoader
import json
import os
from mole.lm.LMConfig import LMConfig
from mole.lm.model import MoLELM
from mole.lm.dataset import PretrainDataset
from transformers import AutoTokenizer
import argparse
from mole.lm.scheduler import LinearWarmupCosineAnnealingLR

arf = argparse.ArgumentParser()
arfr = arf.add_argument_group('required arguments')
arfr.add_argument('--model_config_path', type=str, required=True, help='path to config file')
args, _ = arf.parse_known_args()

@contextmanager
def _timer(name: str, timing_raw):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log/dist_training"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

pwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(pwd, args.model_config_path), 'r') as f:
    config = json.load(f)
    config['max_steps'] = config['max_samples'] // config['batch_size']

if accelerator.is_local_main_process:
    import wandb
    import datetime
    name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(project="mole-lm", name=name, config=config)
    
lmconfig = LMConfig(**config)
model = MoLELM(lmconfig)

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

dataset = PretrainDataset(tokenizer, max_length=config['max_length'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'] // accelerator.num_processes)


optimizer = torch.optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])

if accelerator.is_local_main_process:
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    config['total_params'] = total_params
    config['model_params'] = {k: p.numel() for k, p in model.named_parameters()}
    wandb.config.update(config)

scheduler = LinearWarmupCosineAnnealingLR(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_steps'])

criterion = torch.nn.CrossEntropyLoss(reduction='none')

dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

dim_embed = config['dim'] * (config['n_routed_mole_experts'] * config['n_layers'] + (1 if config['offload_tok_embbedings'] else 0))
dtype = torch.float32
if accelerator.is_local_main_process and dim_embed > 0:
    embedding = SparseEmbedding(
                    config['vocab_size'], 
                    dim_embed, 
                    optimizer_params = {
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "weight_decay": config['weight_decay'],
                        "eps": 1e-8,
                    },
                    std = config['embedding_init_std'],
                    output_dtype = dtype,
                    optimizer_type= config['optimizer_type'] if 'optimizer_type' in config else 'adamw',
                )
    config['embedding_params'] = sum(p.numel() for p in embedding.parameters())
    wandb.config.update(config)

with accelerator.profile() if config['is_profile'] else nullcontext() as prof:
    for step, (x, y, mask) in enumerate(dataloader):
        if step >= config['max_steps']:
            break
        time_raw = {}

        if dim_embed > 0:
            with _timer('embedding', time_raw) if accelerator.is_local_main_process else nullcontext():
                if accelerator.num_processes > 1:
                    gather_list = [torch.empty(x.shape, device=accelerator.device, dtype=torch.int32) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
                    dist.gather(x.to(torch.int32), gather_list, dst=0)
                    if accelerator.is_local_main_process:
                        indexs = torch.stack(gather_list, dim=0)
                        embeds = embedding(indexs)
                        scatter_list = [embeds[_] for _ in range(embeds.shape[0])]
                    else:
                        scatter_list = None

                    embed = torch.empty(x.shape + (dim_embed,), device=accelerator.device, dtype=dtype)
                    dist.scatter(embed, scatter_list, src=0)
                    if accelerator.is_local_main_process:
                        del embeds, indexs, gather_list, scatter_list
                else:
                    embed = embedding(x)

                embed.requires_grad_(True)
                embed.retain_grad()
        else:
            embed = None

        with _timer('forward', time_raw) if accelerator.is_local_main_process else nullcontext():
            out = model(input_ids=x, embedding_input=embed)
            logits = out.logits
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.shape)
            loss = (loss * mask).sum() / mask.sum()

        with _timer('backward_and_update', time_raw) if accelerator.is_local_main_process else nullcontext():
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()


        if dim_embed > 0:
            with _timer('embedding_update', time_raw) if accelerator.is_local_main_process else nullcontext():
                # 如果进程数量>1
                if accelerator.num_processes > 1:
                    gather_list = [torch.empty_like(embed.grad) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
                    dist.gather(embed.grad, gather_list, dst=0)
                if accelerator.is_local_main_process:
                    if accelerator.num_processes > 1:
                        grad = torch.cat(gather_list, dim=0)
                    else:
                        grad = embed.grad
                    lr = optimizer.param_groups[0]['lr']
                    embedding.apply_gradients(output_grad = grad, lr = lr)

        scheduler.step(step) # 重要 https://github.com/huggingface/accelerate/issues/2142
        
        if accelerator.is_local_main_process:
            wandb.log({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                }, step=step)
            for k, v in time_raw.items():
                wandb.log({f'timing/{k}': v}, step=step)
            # 报告每个参数的grad的norm
            for name, p in accelerator.unwrap_model(model).named_parameters():
                if p.grad is not None:
                    wandb.log({f'grad_norm/{name}': p.grad.norm().item()}, step=step)
            if dim_embed > 0:
                wandb.log({'grad_norm/embedding': embed.grad.norm().item()})

        
    
    if accelerator.is_local_main_process:
            unwarpped_model = accelerator.unwrap_model(model)
            model_state_dict = unwarpped_model.state_dict()
            embedding_state_dict = embedding.state_dict() if dim_embed > 0 else {}
            state_dict = {
                **model_state_dict,
                **embedding_state_dict,
            }
            os.makedirs(f"/{pwd}/{name}", exist_ok=True)
            torch.save(state_dict, f"/{pwd}/{name}/checkpoint.pt")

