from accelerate import Accelerator
import torch.distributed as dist
import os
import torch
from embedding_offload.embedding_adamw import SparseEmbedding
from accelerate import ProfileKwargs

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

if accelerator.is_local_main_process:
    embedding = SparseEmbedding(1000, 3, optimizer_params = {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            })
    indexs = torch.randint(0, 1000, (2, 200, 40000))
with accelerator.profile() as prof:
    for _ in range(10):
        if accelerator.is_local_main_process:
            embeds = embedding(indexs)
            # 按第0维切分
            scatter_list = torch.split(embeds, 1, dim=0)
        else:
            scatter_list = None

        # out = torch.empty(3, device=accelerator.device)
        # dist.scatter(out, scatter_list, src=0)
        out = [None]
        dist.scatter_object_list(out, scatter_list, src=0) # 太慢了

        embed = out[0].to(accelerator.device)
        prof.step()
        # profiler.export_chrome_trace('/NAS/wujunkang/guizhiyu/mole/dist/trace.json')  # 保存日志以供 Chrome Tracing 可视化

# print(accelerator.device, out)