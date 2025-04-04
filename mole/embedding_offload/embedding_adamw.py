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
                 ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.optimizer_params = optimizer_params
        self.device = device
        self.optim_device = optim_device
        self.output_dtype = output_dtype
        # 初始化权重在CPU上
        self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, pin_memory=True), requires_grad=False)
        # self.weight = torch.empty((num_embeddings, embedding_dim), device=device, pin_memory=True)
        nn.init.normal_(self.weight, std=std)
        
        self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
        self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
        self.state_steps = 0


    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        self.indices = indices.to(self.device).view(-1)
        output_shape = indices.shape + (self.embedding_dim,)
        output = self.weight[self.indices].to(device=indices.device, dtype=self.output_dtype).detach()
        return output.view(output_shape)
    @torch.no_grad()
    def apply_gradients(self, output_grad: torch.Tensor = None, lr: float = None):
        output_grad = output_grad.view(-1, self.embedding_dim)
        unique_indices, inverse = torch.unique(self.indices, return_inverse=True)

        grad = torch.zeros((unique_indices.shape[0], self.embedding_dim), device=output_grad.device, dtype=output_grad.dtype)
        grad.index_add_(0, inverse.to(output_grad.device), output_grad)
        grad = grad.to(self.optim_device, dtype=torch.float32, non_blocking=True)
        param = self.weight[unique_indices].to(self.optim_device, non_blocking=True)
        exp_avg = self.exp_avgs[unique_indices].to(self.optim_device, non_blocking=True)
        exp_avg_sq = self.exp_avg_sqs[unique_indices].to(self.optim_device, non_blocking=True)

        # calc

        self.state_steps += 1
        lr = lr or self.optimizer_params["lr"]
        beta1, beta2, weight_decay, eps = self.optimizer_params["beta1"], self.optimizer_params["beta2"], self.optimizer_params["weight_decay"], self.optimizer_params["eps"]

        param.mul_(1 - lr * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**self.state_steps
        bias_correction2 = 1 - beta2**self.state_steps

        step_size_neg = - lr / bias_correction1

        denom = (exp_avg_sq.sqrt() / (sqrt(bias_correction2) * step_size_neg)).add_(eps / step_size_neg)
        param.addcdiv_(exp_avg, denom)

        # write back
        self.weight[unique_indices] = param.to(self.device, non_blocking=True)
        self.exp_avgs[unique_indices] = exp_avg.to(self.device, non_blocking=True)
        self.exp_avg_sqs[unique_indices] = exp_avg_sq.to(self.device, non_blocking=True)






            
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