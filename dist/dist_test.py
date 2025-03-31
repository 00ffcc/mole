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
    embedding = SparseEmbedding(10000, 300, optimizer_params = {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            })
    indexs = torch.randint(0, 10000, (2, 20, 4000), device=accelerator.device)
# with torch.profiler.profile(
#             activities=[
#                 torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
#             schedule=torch.profiler.schedule(
#                 wait=1,  # 前1步不采样
#                 warmup=1,  # 第2步作为热身，不计入结果
#                 active=3,  # 采集后面3步的性能数据
#                 repeat=2),  # 重复2轮
#             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # 保存日志以供 TensorBoard 可视化
#             record_shapes=True,  # 记录输入张量的形状
#             profile_memory=True,  # 分析内存分配
#             with_stack=True  # 记录操作的调用堆栈信息
#         ) as prof:
with accelerator.profile() as prof:
    for _ in range(10):
        if accelerator.is_local_main_process:
            embeds = embedding(indexs)
            # 按第0维切分
            scatter_list = [embeds[0], embeds[1]]
        else:
            scatter_list = None

        out = torch.empty((20, 4000, 300), device=accelerator.device)
        dist.scatter(out, scatter_list, src=0)
        # out = [None]
        # dist.scatter_object_list(out, scatter_list, src=0) # 太慢了
        # out = out[0].to(accelerator.device)

        out.requires_grad_(True)
        out.retain_grad()
        loss = out.sum()
        loss.backward()
        if accelerator.is_local_main_process:
            # print(out.grad)
            print(embedding.weight.grad)
            # torch.distributed.
        dist.barrier()
        prof.step()
        # profiler.export_chrome_trace('/NAS/wujunkang/guizhiyu/mole/dist/trace.json')  # 保存日志以供 Chrome Tracing 可视化

# print(accelerator.device, out)