import torch
import torch.nn as nn
from mole.kernel.index_to import index_to_pinned, index_to_cuda
class SparseEmbedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 optimizer_params: dict,
                 device: str = "cpu",
                 optim_device: str = "cpu",
                 std: float = 0.01,
                 optimizer_type: str = "adamw",
                 **kwargs,
                 ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.optimizer_params = optimizer_params
        self.device = device
        self.optim_device = optim_device
        self.optimizer_type = optimizer_type
        # 初始化权重在CPU上
        self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, pin_memory=True), requires_grad=False)
        nn.init.normal_(self.weight, std=std)

        self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
        self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
    @torch.no_grad()
    def forward(self, indices: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        self.indices = indices.view(-1)
        output_shape = indices.shape + (self.embedding_dim,)
        return index_to_cuda(self.weight, self.indices, dtype=dtype).view(output_shape)
    @torch.no_grad()
    def apply_gradients(self, output_grad: torch.Tensor = None, lr: float = None):
        output_grad = output_grad.view(-1, self.embedding_dim).to(torch.float32)
        unique_indices, inverse = torch.unique(self.indices, return_inverse=True)

        grad = torch.zeros((unique_indices.shape[0], self.embedding_dim), device=output_grad.device, dtype=torch.float32)
        grad.index_add_(0, inverse.to(output_grad.device), output_grad)

        param = index_to_cuda(self.weight, unique_indices)
        exp_avg = index_to_cuda(self.exp_avgs, unique_indices)
        exp_avg_sq = index_to_cuda(self.exp_avg_sqs, unique_indices)
        
        lr = lr if lr is not None else self.optimizer_params["lr"]
        beta1, beta2, weight_decay, eps = self.optimizer_params["beta1"], self.optimizer_params["beta2"], self.optimizer_params["weight_decay"], self.optimizer_params["eps"]

        param.mul_(1 - lr * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(exp_avg * (-lr), denom)

        # write back
        index_to_pinned(self.weight, unique_indices, param)
        index_to_pinned(self.exp_avgs, unique_indices, exp_avg)
        index_to_pinned(self.exp_avg_sqs, unique_indices, exp_avg_sq)

if __name__ == "__main__":
    embed = SparseEmbedding(10, 4, optimizer_params = {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "eps": 1e-8,
    },
    )
    
    
    # 模拟训练步骤
    indices = torch.randint(0, 10, (1, 12), dtype=torch.int32).cuda()
    out = embed(indices)
    loss = out.sum()
    # loss.backward()
    grad = torch.ones_like(out)
    embed.apply_gradients(output_grad=grad, lr=1e-3)