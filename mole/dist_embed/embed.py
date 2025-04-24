import torch
import torch.nn as nn
from mole.kernel.index_to import index_to_cuda, index_to_kernel
import torch.distributed as dist

class SparseEmbedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 optimizer_params: dict,
                 device: str = "cpu",
                 optim_device: str = "cpu",
                 std: float = 0.01,
                 optimizer_type: str = "adamw",
                 accelerator=None,
                 src_rank=0,
                 **kwargs,
                 ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.optimizer_params = optimizer_params
        self.device = device
        self.optim_device = optim_device
        self.optimizer_type = optimizer_type
        self.accelerator = accelerator
        self.src_rank = src_rank
        if accelerator.num_processes == 1 or src_rank == accelerator.process_index:
            self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, pin_memory=True), requires_grad=False)
            nn.init.normal_(self.weight, std=std)

            self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
            self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
    @torch.no_grad()
    def forward(self, indices: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.accelerator.num_processes > 1:
            gather_list = [torch.empty(indices.shape, device=self.accelerator.device, dtype=torch.int32) for _ in range(self.accelerator.num_processes)] if self.src_rank == self.accelerator.process_index else None
            dist.gather(indices.to(torch.int32), gather_list, dst=0)
            if self.src_rank == self.accelerator.process_index:
                indexs = torch.stack(gather_list, dim=0)
                embeds = self.forward_single(indexs, dtype=dtype)
                scatter_list = [embeds[_] for _ in range(embeds.shape[0])]
            else:
                scatter_list = None

            embed = torch.empty(indices.shape + (self.embedding_dim,), device=self.accelerator.device, dtype=dtype)
            dist.scatter(embed, scatter_list, src=0)
            if self.src_rank == self.accelerator.process_index:
                del embeds, indexs, gather_list, scatter_list
        else:
            embed = self.forward_single(indices, dtype=dtype)

        embed.requires_grad_(True)
        embed.register_hook(self.apply_gradients_hook)
        return embed
    
    @torch.no_grad()
    def forward_single(self, indices: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        self.indices = indices.view(-1)
        output_shape = indices.shape + (self.embedding_dim,)
        return index_to_cuda(self.weight, self.indices, dtype=dtype).view(output_shape)
    
    @torch.no_grad()
    def apply_gradients_hook(self, grad: torch.Tensor = None):
        if self.accelerator.num_processes > 1:
            dist.barrier()
            torch.cuda.synchronize()
            gather_list = [torch.empty_like(grad) for _ in range(self.accelerator.num_processes)] if self.src_rank == self.accelerator.process_index else None
            dist.gather(grad.contiguous(), gather_list, dst=self.src_rank) # https://github.com/pytorch/pytorch/issues/73515 gather的输入必须是 contiguous
            if self.src_rank == self.accelerator.process_index:
                grad = torch.cat(gather_list, dim=0)
                self.apply_gradients_single(grad)
                del grad, gather_list
            else:
                del grad
        else:
            self.apply_gradients_single(grad)

    @torch.no_grad()
    def apply_gradients_single(self, output_grad: torch.Tensor = None):
        output_grad = output_grad.view(-1, self.embedding_dim).to(torch.float32)
        unique_indices, inverse = torch.unique(self.indices, return_inverse=True)

        grad = torch.zeros((unique_indices.shape[0], self.embedding_dim), device=output_grad.device, dtype=torch.float32)
        grad.index_add_(0, inverse.to(output_grad.device), output_grad)
        
        lr = self.optimizer_params["lr"]
        beta1, beta2, weight_decay, eps = self.optimizer_params["beta1"], self.optimizer_params["beta2"], self.optimizer_params["weight_decay"], self.optimizer_params["eps"]

        index_to_kernel.adamw(
            self.weight,
            grad,
            self.exp_avgs,
            self.exp_avg_sqs,
            unique_indices,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
        )
    
    def update_lr(self, lr: float):
        self.optimizer_params["lr"] = lr

if __name__ == "__main__":
    from accelerate import Accelerator
    accelerator = Accelerator()
    embed = SparseEmbedding(10, 4, 
        optimizer_params = {
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
        accelerator=accelerator,
    )
    
    # 模拟训练步骤
    indices = torch.randint(0, 10, (1, 3), dtype=torch.int32).cuda()
    out = embed(indices)
    loss = out.sum()
    # loss.backward()
    accelerator.backward(loss)