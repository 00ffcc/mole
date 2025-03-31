from accelerate import Accelerator, ProfileKwargs
import torch.distributed as dist
import torch
from embedding_offload.embedding_adamw import SparseEmbedding

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log/dist"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

n = 10000
dim_embed = 300
B = 20
L = 4000

if accelerator.is_local_main_process:
    embedding = SparseEmbedding(n, dim_embed, optimizer_params = {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            })
    indexs = torch.randint(0, n, (B * dist.get_world_size(), L), device=accelerator.device)
with accelerator.profile() as prof:
    for _ in range(10):
        if accelerator.is_local_main_process:
            embeds = embedding(indexs)
            # 按第0维切分, 每B个数据送给一个进程
            scatter_list = list(torch.split(embeds, B, dim=0))
        else:
            scatter_list = None

        out = torch.empty((B, L, dim_embed), device=accelerator.device)
        dist.scatter(out, scatter_list, src=0)

        out.requires_grad_(True)
        out.retain_grad()
        loss = out.sum()
        loss.backward()
        gather_list = [torch.empty_like(out.grad) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
        dist.gather(out.grad, gather_list, dst=0)
        if accelerator.is_local_main_process:
            grad = torch.cat(gather_list, dim=0)
            embedding.apply_gradients(grad)
        # dist.barrier()
        prof.step()
