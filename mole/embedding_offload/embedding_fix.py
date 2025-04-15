import torch
import torch.nn as nn
from math import sqrt
class SparseEmbedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 optimizer_params: dict,
                 device: str = "cpu",
                 optim_device: str = "cpu",
                 output_dtype: torch.dtype = torch.float32,
                 std: float = 0.01,
                 **kwargs,
                 ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.optimizer_params = optimizer_params
        self.device = device
        self.optim_device = optim_device
        self.output_dtype = output_dtype

        # 初始化权重在CPU上
        self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, embedding_dim), device=device), requires_grad=False)
        nn.init.normal_(self.weight, std=std)
        
        self.exp_avgs = torch.zeros_like(self.weight, device='cpu')
        self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu')

        self.lr = 0.0
        self.global_step = 0
        self.step = torch.zeros((num_embeddings, 1), device='cpu')

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        output_shape = indices.shape + (self.embedding_dim,)
        output_device = indices.device
        indices = indices.to(self.device).view(-1)
        self.unique_indices, self.inverse = torch.unique(indices, return_inverse=True)

        self.global_step += 1
        t = self.optimizer_params["beta1"] / sqrt(self.optimizer_params["beta2"])
        sp = self.global_step - self.step[self.unique_indices]
        self.step[self.unique_indices] = self.global_step

        self.weight[self.unique_indices] -= self.lr * ((t**sp - 1) / (t - 1)) * (self.exp_avgs[self.unique_indices] / (torch.sqrt(self.exp_avg_sqs[self.unique_indices]) + self.optimizer_params["eps"]))
        self.exp_avgs[self.unique_indices] *= self.optimizer_params["beta1"] ** sp
        self.exp_avg_sqs[self.unique_indices] *= self.optimizer_params["beta2"] ** sp

        output = self.weight[indices].to(device=output_device, dtype=self.output_dtype).detach()
        return output.view(output_shape)
    @torch.no_grad()
    def apply_gradients(self, output_grad: torch.Tensor = None, lr: float = None):
        output_grad = output_grad.view(-1, self.embedding_dim)
        

        grad = torch.zeros((self.unique_indices.shape[0], self.embedding_dim), device=output_grad.device, dtype=output_grad.dtype)
        grad.index_add_(0, self.inverse.to(output_grad.device), output_grad)
        grad = grad.to(self.optim_device, dtype=torch.float32, non_blocking=True)
        exp_avg = self.exp_avgs[self.unique_indices].to(self.optim_device, non_blocking=True)
        exp_avg_sq = self.exp_avg_sqs[self.unique_indices].to(self.optim_device, non_blocking=True)
        
        self.lr = lr or self.optimizer_params["lr"]

        beta1, beta2 = self.optimizer_params["beta1"], self.optimizer_params["beta2"]

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # write back
        self.exp_avgs[self.unique_indices] = exp_avg.to(self.device, non_blocking=True)
        self.exp_avg_sqs[self.unique_indices] = exp_avg_sq.to(self.device, non_blocking=True)

            
if __name__ == "__main__":
    embed = SparseEmbedding(10, 4, optimizer_params = {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "eps": 1e-8,
    },
    output_dtype=torch.float16,)
    
    
    # 模拟训练步骤
    indices = torch.randint(0, 10, (1, 12)).cuda()
    out = embed(indices)
    loss = out.sum()
    loss.backward()
    embed.apply_gradients()
    print(embed.state_steps)