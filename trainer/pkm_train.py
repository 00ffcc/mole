from accelerate import Accelerator, ProfileKwargs
import torch
from contextlib import nullcontext, contextmanager
from codetiming import Timer
from torch.utils.data import DataLoader
import json
import os
from mole.lm.pkm_lm import PKMLM, pkmlm_config
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
        torch.cuda.synchronize() # TODO
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
    config['embedding_output_dtype'] = torch.bfloat16 if accelerator.mixed_precision else torch.float32

    
lmconfig = pkmlm_config(**config)
model = PKMLM(lmconfig)

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

dataset = PretrainDataset(tokenizer, max_length=config['max_length'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'] // accelerator.num_processes)


optimizer = torch.optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])

if accelerator.is_local_main_process:
    # 统计参数量
    activated_params = sum(p.numel() for p in model.parameters())
    config['activated_params'] = activated_params
    config['model_params'] = {k: p.numel() for k, p in model.named_parameters()}
    offloaded_params = 0
    if config['offload_tok_embbedings']:
        offloaded_params += sum(p.numel() for p in model.tok_embeddings[0].parameters())
    for m in model.pkm_layers:
        if not config['use_lucidrains_pkm']:
            offloaded_params += sum(p.numel() for p in m.values[0].parameters())
    config['offloaded_params'] = offloaded_params

    import wandb
    import datetime
    name = f"{config['norm_type']}-a{activated_params//1000000}m-o{offloaded_params//1000000}m-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(project="pkm-lm", name=name, config=config)

scheduler = LinearWarmupCosineAnnealingLR(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_steps'])

criterion = torch.nn.CrossEntropyLoss(reduction='none')

dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

is_timing = accelerator.is_local_main_process and config.get('is_timing', True)
time_raw = {}
with accelerator.profile() if config['is_profile'] else nullcontext() as prof:
    for step, (x, y, mask) in enumerate(dataloader):
        with _timer('step', time_raw) if is_timing else nullcontext():
            if step >= config['max_steps']:
                break

            with _timer('forward', time_raw) if is_timing else nullcontext():
                out = model(input_ids=x)
                logits = out.logits
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.shape)
                loss = (loss * mask).sum() / mask.sum()

            with _timer('backward_and_update', time_raw) if is_timing else nullcontext():
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                lr = optimizer.param_groups[0]['lr']
                model.update_lr(lr)

            scheduler.step(step) # 重要 https://github.com/huggingface/accelerate/issues/2142
            
            if accelerator.is_local_main_process:
                wandb.log({
                        'loss': loss.item(),
                        'config/lr': optimizer.param_groups[0]['lr'],
                    }, step=step)
                for k, v in time_raw.items():
                    wandb.log({f'timing/{k}': v}, step=step)

                # for name, p in accelerator.unwrap_model(model).named_parameters():
                #     if p.grad is not None:
                #         wandb.log({f'grad_norm/{name}': p.grad.norm().item()}, step=step)

        
    
if accelerator.is_local_main_process:
    unwarpped_model = accelerator.unwrap_model(model)
    model_state_dict = unwarpped_model.state_dict()
    # embedding_state_dict = embedding.state_dict() if dim_embed > 0 else {}
    state_dict = {
        **model_state_dict,
        # **embedding_state_dict,
    }
    os.makedirs(f"/{pwd}/{name}", exist_ok=True)
    torch.save(state_dict, f"/{pwd}/{name}/ckpt.pt")

