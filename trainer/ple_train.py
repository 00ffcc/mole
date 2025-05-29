from accelerate import Accelerator, ProfileKwargs
import torch
from contextlib import nullcontext, contextmanager
from codetiming import Timer
from torch.utils.data import DataLoader
import json
import yaml
import os
from mole.lm.ple_model import PLELM, plelm_config
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
    if args.model_config_path.endswith('.json'):
        config = json.load(f)
    elif args.model_config_path.endswith('.yaml'):
        config = yaml.full_load(f)
    else:
        raise ValueError('Unsupported config file type')
    config['max_steps'] = config['max_samples'] // config['batch_size']
    config['embedding_output_dtype'] = torch.bfloat16 if accelerator.mixed_precision else torch.float32

    
lmconfig = plelm_config(**config)
model = PLELM(lmconfig)

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

dataset = PretrainDataset(tokenizer, max_length=config['max_length'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'] // accelerator.num_processes)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])

if accelerator.is_local_main_process:
    config['activated_params'] = sum(p.numel() for p in model.parameters())
    config['model_params'] = {k: p.numel() for k, p in model.named_parameters()}
    embedding_dim = len(lmconfig.ple_layer_ids) * lmconfig.ple_dim + (lmconfig.dim if lmconfig.offload_tok_embbedings else 0)
    config['offloaded_params'] = embedding_dim * lmconfig.vocab_size
    config['embedding_dim'] = embedding_dim

    import wandb
    import datetime
    name = f"{config['norm_type']}-a{config['activated_params']//1000000}m-o{config['offloaded_params']//1000000}m-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(project="ple-lm", name=name, config=config)

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

