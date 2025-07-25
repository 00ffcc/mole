import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import torch.distributed as dist
import os
from types import SimpleNamespace
import math

pwd = os.path.dirname(os.path.abspath(__file__))

index_to_kernel = load(name='index_to_cuda', 
                            sources=[
                                os.path.join(pwd, '../kernel/index_to.cpp'), 
                                os.path.join(pwd, '../kernel/index_to_kernel.cu')
                                ], 
                            verbose=False,
                            extra_cflags=['-O3'],
                            extra_cuda_cflags=['-O3'],
                            )
def calc_local_dim(embedding_dim, world_size, local_rank):
    rank_dim = math.ceil(embedding_dim / world_size)
    local_dim = rank_dim if local_rank != world_size - 1 else embedding_dim - rank_dim * (world_size - 1)
    return local_dim

class SparseEmbeddingFunc(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, 
                indices: torch.Tensor, 
                weight: torch.Tensor | None,
                exp_avgs: torch.Tensor | None,
                exp_avg_sqs: torch.Tensor | None,
                config,
                ) -> torch.Tensor:
        '''
        indices: (..., N)
        weight: (NE, LC)
        output: (..., C)
        output = weight[indices, :]
        '''
        ctx.config = config

        if not config.ddp:
            return SparseEmbeddingFunc.forward_single(ctx, indices, weight, exp_avgs, exp_avg_sqs, config)

        indices_list = [torch.empty_like(indices) for _ in range(config.world_size)]

        dist.all_gather(indices_list, indices.contiguous())
        indices = torch.stack(indices_list, dim=0)
        embeds = SparseEmbeddingFunc.forward_single(ctx, indices, weight, exp_avgs, exp_avg_sqs, config)
        scatter_list = [embeds[_] for _ in range(embeds.shape[0])]

        output_list = [torch.empty(
            indices.shape[1:] + (calc_local_dim(config.embedding_dim, config.world_size, rank),), 
            dtype=config.dtype, 
            device=indices.device,
        ) for rank in range(config.world_size)]

        dist.all_to_all(output_list, scatter_list)
        embed = torch.cat(output_list, dim=-1)
        return embed


    @staticmethod
    @torch.no_grad()
    def forward_single(ctx, 
                indices: torch.Tensor, 
                weight: torch.Tensor,
                exp_avgs: torch.Tensor,
                exp_avg_sqs: torch.Tensor,
                config,
                ) -> torch.Tensor:
        '''
        indices: (..., N)
        weight: (NE, C)
        output: (..., C)
        output = weight[indices, :]
        '''
        unique_indices, inverse = torch.unique(indices.view(-1), sorted=True, return_inverse=True)
        unique_indices = unique_indices.to(torch.int32)
        ctx.save_for_backward(weight, exp_avgs, exp_avg_sqs, unique_indices, inverse)
        embeds = torch.empty(unique_indices.shape + (weight.shape[-1],), dtype=config.dtype, device=unique_indices.device)
        index_to_kernel.index_to_cuda(weight, unique_indices, embeds)
        embeds = embeds[inverse].view(indices.shape + (weight.shape[-1],))
        return embeds

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor):
        '''
        grad_output: (..., C)
        '''
        C = grad_output.shape[-1]
        grad_output = grad_output.view(-1, C)
        config = ctx.config
        if not config.ddp:
            SparseEmbeddingFunc.backward_single(ctx, grad_output)
            return None, None, None, None, None

        scatter_list = [i.contiguous() for i in grad_output.split(config.split_list, dim=-1)]
        
        output_list = [torch.empty(
            grad_output.shape[0], 
            calc_local_dim(config.embedding_dim, config.world_size, config.local_rank),
            device=grad_output.device,
            dtype=grad_output.dtype,
        ) for _ in range(config.world_size)]

        dist.all_to_all(output_list, scatter_list)
        grad = torch.cat(output_list, dim=0)
        SparseEmbeddingFunc.backward_single(ctx, grad)
        
        return None, None, None, None, None


    @staticmethod
    @torch.no_grad()
    def backward_single(ctx, grad_output: torch.Tensor):
        '''
        grad_output: (..., C)
        '''
        C = grad_output.shape[-1]
        config = ctx.config
        weight, exp_avgs, exp_avg_sqs, unique_indices, inverse = ctx.saved_tensors
        unique_grad = torch.zeros((unique_indices.shape[0], C), device=grad_output.device, dtype=grad_output.dtype)
        unique_grad.index_add_(0, inverse, grad_output.view(-1, C))
        unique_grad = SparseEmbeddingFunc.grad_clip(unique_grad, config.grad_clip_max_norm)
        
        index_to_kernel.adamw(
            weight,
            unique_grad,
            exp_avgs,
            exp_avg_sqs,
            unique_indices,
            config.lr,
            config.beta1,
            config.beta2,
            config.weight_decay,
            config.eps,
        )


    @staticmethod
    @torch.no_grad()
    def grad_clip(grad, max_norm):
        if max_norm is None:
            return grad
        grad_norm = torch.norm(grad, p=2, dim=-1)
        scale = torch.clamp(max_norm / (grad_norm + 1e-6), max=1.0)
        grad = grad * scale.unsqueeze(-1)
        return grad
        



class SparseEmbedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 optimizer_params: dict,
                 std: float = 0.01,
                 embedding_output_dtype: torch.dtype | str = torch.float32,
                 params_dtype: torch.dtype | str = torch.float32,
                 **kwargs,
                 ):
        super().__init__()

        str2dtype = {
            "float32" : torch.float32,
            "bfloat16": torch.bfloat16,
            "fp32"    : torch.float32,
            "bf16"    : torch.bfloat16,
        }
        if isinstance(embedding_output_dtype, str):
            if embedding_output_dtype.lower() not in str2dtype:
                raise ValueError(f"Unsupported dtype: {embedding_output_dtype}")
            embedding_output_dtype = str2dtype[embedding_output_dtype.lower()]
        if isinstance(params_dtype, str):
            if params_dtype.lower() not in str2dtype:
                raise ValueError(f"Unsupported dtype: {params_dtype}")
            params_dtype = str2dtype[params_dtype.lower()]

        if dist.is_available() and dist.is_initialized():
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
            ddp = world_size > 1
            local_dim = calc_local_dim(embedding_dim, world_size, local_rank)
        else:
            local_rank = 0
            world_size = 1
            ddp = False
            local_dim = embedding_dim

        self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, local_dim), device='cpu', pin_memory=True, requires_grad=True, dtype=params_dtype))
        nn.init.normal_(self.weight, std=std, generator=torch.Generator().manual_seed(local_rank))

        self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True, dtype=params_dtype)
        self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True, dtype=params_dtype)
        
        self.config = SimpleNamespace(
            embedding_dim=embedding_dim,
            local_dim=local_dim,
            local_rank=local_rank,
            lr=optimizer_params.get("lr", 1e-3),
            beta1=optimizer_params.get("beta1", 0.9),
            beta2=optimizer_params.get("beta2", 0.999),
            weight_decay=optimizer_params.get("weight_decay", 0.01),
            eps=optimizer_params.get("eps", 1e-8),
            grad_clip_max_norm=optimizer_params.get("grad_clip_max_norm", None),
            ddp=ddp,
            dtype=embedding_output_dtype,
            world_size=world_size,
            split_list=[calc_local_dim(embedding_dim, world_size, i) for i in range(world_size)]
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return SparseEmbeddingFunc.apply(indices, self.weight, self.exp_avgs, self.exp_avg_sqs, self.config)
    

if __name__ == "__main__":
    from accelerate import Accelerator
    accelerator = Accelerator()
    NE = 10
    C = 2
    B = 10
    N = 4
    embed = SparseEmbedding(NE, C, 
                    optimizer_params = {
                        "lr": 1e-3,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "weight_decay": 0.00,
                        "eps": 0,
                    },
                )
    for i in range(NE):
        embed.weight[i, 0]=i
    res = embed(torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.int32, device=accelerator.device))
    print(res)
    res.backward(torch.ones_like(res))
    res = embed(torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.int32, device=accelerator.device))
    print(res)
