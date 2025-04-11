from accelerate import Accelerator, ProfileKwargs
import torch.distributed as dist
import torch
from mole.embedding_offload.embedding_adamw import SparseEmbedding
import time
profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log/dist_training"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
dim_embed = 9984
if accelerator.is_local_main_process and dim_embed > 0:
    embedding = SparseEmbedding(
                    50304, 
                    dim_embed, 
                    optimizer_params = {
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "weight_decay": 0.0001,
                        "eps": 1e-8,
                    },
                    std = 0.01,
                    output_dtype = torch.bfloat16)
with accelerator.profile():
    for i in range(10):
        ti = time.time()
        x = torch.randint(0, 50000, (4, 2048), device=accelerator.device)
        gather_list = [torch.empty_like(x) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
        dist.gather(x, gather_list, dst=0)
        if accelerator.is_local_main_process:
            indexs = torch.stack(gather_list, dim=0)
            embeds = embedding(indexs)
            scatter_list = [embeds[_] for _ in range(embeds.shape[0])]
        else:
            scatter_list = None

        embed = torch.empty(x.shape + (dim_embed,), device=accelerator.device, dtype=torch.bfloat16)
        dist.scatter(embed, scatter_list, src=0)
        if accelerator.is_local_main_process:
            del embeds, indexs, gather_list, scatter_list

        embed.requires_grad_(True)
        embed.retain_grad()
        loss = embed.sum()
        accelerator.backward(loss)

        gather_list = [torch.empty_like(embed.grad) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
        dist.gather(embed.grad, gather_list, dst=0)
        if accelerator.is_local_main_process:
            grad = torch.cat(gather_list, dim=0)
            embedding.apply_gradients(output_grad = grad, lr = 0.0001)
        
        accelerator.wait_for_everyone()
        accelerator.print(time.time()-ti)
