import torch
from embedding_offload.embedding_adamw import SparseEmbedding

if __name__ == "__main__":
    embed = SparseEmbedding(1000, 1024, optimizer_params = {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            })
    input_tensor = torch.randint(0, 1000, (10, 128))
    
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
            schedule=torch.profiler.schedule(
                wait=1,  # 前1步不采样
                warmup=1,  # 第2步作为热身，不计入结果
                active=3,  # 采集后面3步的性能数据
                repeat=2),  # 重复2轮
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # 保存日志以供 TensorBoard 可视化
            record_shapes=True,  # 记录输入张量的形状
            profile_memory=True,  # 分析内存分配
            with_stack=True  # 记录操作的调用堆栈信息
        ) as profiler:
        for _ in range(10):
            out = embed(input_tensor)
            out.sum().backward()
            embed.apply_gradients()
            profiler.step()