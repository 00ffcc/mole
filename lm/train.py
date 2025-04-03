from accelerate import Accelerator, ProfileKwargs
import torch.distributed as dist
import torch
from embedding_offload.embedding_adamw import SparseEmbedding
from contextlib import nullcontext
from torch.utils.data import DataLoader
import json
import os
from .LMConfig import LMConfig
from .model import MoLELM
from .dataset import PretrainDataset
from transformers import AutoTokenizer

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log/dist_training"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

pwd = os.path.dirname(os.path.abspath(__file__))
with open(f'{pwd}/config/config.json') as f:
    config = json.load(f)

if accelerator.is_local_main_process:
    import wandb
    import datetime
    name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(project="mole-lm", name=name, config=config)
    
lmconfig = LMConfig(**config)
model = MoLELM(lmconfig)

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

dataset = PretrainDataset(config['data_path'], tokenizer, max_length=config['max_length'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / config['warmup_steps'] if epoch < config['warmup_steps'] else 1.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40000, T_mult=2, eta_min=0)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

dataloader, model, optimizer, scheduler, warmup_scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler, warmup_scheduler)

dim_embed = config['dim'] * (config['n_routed_mole_experts'] * config['n_layers'] + (1 if config['offload_tok_embbedings'] else 0))
if accelerator.is_local_main_process and dim_embed > 0:
    embedding = SparseEmbedding(config['vocab_size'], dim_embed, optimizer_params = {
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            })

with accelerator.profile() if config['is_profile'] else nullcontext() as prof:
    for step, (x, y, mask) in enumerate(dataloader):
        if dim_embed > 0:
            gather_list = [torch.empty_like(x) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
            dist.gather(x, gather_list, dst=0)
            if accelerator.is_local_main_process:
                indexs = torch.stack(gather_list, dim=0)
                embeds = embedding(indexs)
                scatter_list = [embeds[_] for _ in range(embeds.shape[0])]
            else:
                scatter_list = None

            embed = torch.empty(x.shape + (dim_embed,), device=accelerator.device)
            dist.scatter(embed, scatter_list, src=0)
            if accelerator.is_local_main_process:
                del embeds, indexs, gather_list, scatter_list

            embed.requires_grad_(True)
            embed.retain_grad()
        else:
            embed = None

        out = model(input_ids=x, embedding_input=embed)
        logits = out.logits
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1), weight=mask.view(-1))

        if accelerator.is_local_main_process:
            wandb.log({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                }, step=step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if dim_embed > 0:
            gather_list = [torch.empty_like(embed.grad) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
            dist.gather(embed.grad, gather_list, dst=0)
            if accelerator.is_local_main_process:
                grad = torch.cat(gather_list, dim=0)
                lr = optimizer.param_groups[0]['lr']
                embedding.apply_gradients(grad=grad, lr=lr)

        if step < config['warmup_steps']:
            warmup_scheduler.step()
        else:
            scheduler.step()

