import torch
import torch.nn as nn
from typing import List, Optional, Union
# from torch.optim.adamw import adamw
from embedding_offload.sparse_adamw import adamw
from torch.optim.sgd import SGD
class SparseEmbedding(nn.Module):
    # TODO:
    # 0. nan? 改了啥?
    # 1. unique, []很慢
    # 2. to很慢
    # 3. 如果放cpu上，adamw很慢
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 初始化权重在CPU上
        self.weight = torch.empty((num_embeddings, embedding_dim), device='cpu', pin_memory=True)
        nn.init.normal_(self.weight)
        
        # 初始化AdamW优化器状态
        self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
        self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
        self.state_steps = torch.zeros((num_embeddings, 1), dtype=torch.int32, device='cpu', pin_memory=True)
        # self.state_steps = torch.zeros((1), dtype=torch.int32, device='cuda')

        # 用于临时保存前向传播的激活信息
        self.unique_indices: Optional[torch.Tensor] = None
        self.active_weights: Optional[torch.Tensor] = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # 确保索引在CPU上去重以匹配参数位置
        indices_cpu = indices.cpu().view(-1)
        unique_indices_cpu, inverse = torch.unique(indices_cpu, return_inverse=True)
        unique_indices_cpu = unique_indices_cpu.to(torch.long)
        
        # 保存反向传播所需信息
        self.unique_indices = unique_indices_cpu
        self.inverse = inverse.to(indices.device)
        self.original_shape = indices.shape
        
        # 从CPU参数中提取激活行并传输到GPU
        active_weights = self.weight[unique_indices_cpu].to(indices.device).detach().requires_grad_(True)
        self.active_weights = active_weights
        
        # 构造输出张量
        output_shape = self.original_shape + (self.embedding_dim,)
        output = active_weights[self.inverse].view(output_shape)
        return output

    def apply_gradients(self, 
                        lr: Union[float, torch.Tensor],
                        beta1: float,
                        beta2: float,
                        weight_decay: float,
                        eps: float,
                        optim_device: str,
                        ):
        with torch.no_grad():
            if self.unique_indices is None or self.active_weights.grad is None:
                return
            
            unique_indices = self.unique_indices
            
            # 准备优化器输入
            params = [self.active_weights.to(optim_device, non_blocking=True)]
            grads = [self.active_weights.grad.to(optim_device, non_blocking=True)]
            exp_avgs = [self.exp_avgs[unique_indices].to(optim_device, non_blocking=True)]
            exp_avg_sqs = [self.exp_avg_sqs[unique_indices].to(optim_device, non_blocking=True)]
            state_steps = [self.state_steps[unique_indices].to(optim_device, non_blocking=True)]
            # state_steps = [self.state_steps]
            # 调用AdamW优化函数
            adamw(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                state_steps=state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
            )
            # 写回
            self.weight[unique_indices] = params[0].to(self.weight.device, non_blocking=True)
            self.exp_avgs[unique_indices] = exp_avgs[0].to(self.exp_avgs.device, non_blocking=True)
            self.exp_avg_sqs[unique_indices] = exp_avg_sqs[0].to(self.exp_avg_sqs.device, non_blocking=True)
            self.state_steps[unique_indices] = state_steps[0].to(self.state_steps.device, non_blocking=True)
            # self.state_steps += 1 # 这里与原embedding不同，对没被更新的权重不增加step，得试试哪个效果好，可能不重要。
            # 清理中间变量
            self.unique_indices = None
            self.active_weights = None
            self.inverse = None
            self.original_shape = None

# 示例用法
if __name__ == "__main__":
    embed = SparseEmbedding(10, 4)
    optimizer_params = {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "optim_device": "cuda",
    }
    
    # 模拟训练步骤
    indices = torch.randint(0, 10, (1, 12)).cuda()
    out = embed(indices)
    loss = out.sum()
    loss.backward()
    embed.apply_gradients(**optimizer_params)
    print(embed.state_steps)