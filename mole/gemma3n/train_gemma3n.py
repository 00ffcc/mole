from accelerate import Accelerator, ProfileKwargs
import torch
from contextlib import nullcontext, contextmanager
from codetiming import Timer
from torch.utils.data import DataLoader
import json
import yaml
import os
from mole.gemma3n.modeling_gemma3n import Gemma3nForCausalLM, Gemma3nTextConfig
from mole.lm.dataset import PretrainDataset
from transformers import AutoTokenizer
import argparse
from mole.lm.scheduler import LinearWarmupCosineAnnealingLR
from types import SimpleNamespace

arf = argparse.ArgumentParser()
arf.add_argument('--model_config_path', type=str, required=True, help='path to config file')
args = arf.parse_args()

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
    config = SimpleNamespace(**config)
    
lmconfig = Gemma3nTextConfig(**config.model)
model = Gemma3nForCausalLM(lmconfig)

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

dataset = PretrainDataset(tokenizer, max_length=config.max_length, dataset_name_or_path=config.dataset_name_or_path)
dataloader = DataLoader(dataset, batch_size=config.batch_size_per_device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)

if accelerator.is_local_main_process:
    config.activated_params = sum(p.numel() for p in model.parameters())
    config.model_params = {k: p.numel() for k, p in model.named_parameters()}

    if config.log_backend == 'wandb':
        import wandb as wandb
    elif config.log_backend == 'swanlab':
        import swanlab as wandb
    
    import datetime
    name = f"{config.model.arch}-a{config.activated_params//1000000}m-{datetime.datetime.now().strftime('%m%d%H%M')}"
    wandb.init(project="ple-lm", name=name, config=config.to_dict())

scheduler = LinearWarmupCosineAnnealingLR(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps)

criterion = torch.nn.CrossEntropyLoss(reduction='none')

dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

is_timing = accelerator.is_local_main_process and config.is_timing
time_raw = {}
with accelerator.profile() if config.is_profile else nullcontext() as prof:
    for step, (x, y, mask) in enumerate(dataloader):
        with _timer('step', time_raw) if is_timing else nullcontext():
            with _timer('forward', time_raw) if is_timing else nullcontext():
                out = model(input_ids=x)
                logits = out.logits
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.shape)
                loss = (loss * mask).sum() / mask.sum()

            with _timer('backward_and_update', time_raw) if is_timing else nullcontext():
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            scheduler.step(step) # 重要 https://github.com/huggingface/accelerate/issues/2142
            
            if accelerator.is_local_main_process:
                wandb.log({
                        'loss': loss.item(),
                        'config/lr': optimizer.param_groups[0]['lr'],
                    }, step=step)
                for k, v in time_raw.items():
                    wandb.log({f'timing/{k}': v}, step=step)


