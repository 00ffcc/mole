import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import torch.distributed as dist
import os
from types import SimpleNamespace

pwd = os.path.dirname(os.path.abspath(__file__))

index_to_kernel = load(name='index_to_cuda', 
                            sources=[
                                os.path.join(pwd, '../kernel/embed_bag.cpp'), 
                                os.path.join(pwd, '../kernel/embed_bag.cu')
                                ], 
                            verbose=False,
                            extra_cflags=['-O3'],
                            extra_cuda_cflags=['-O3'],
                            )

class SparseEmbeddingBagFunc(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, 
                scores: torch.Tensor, 
                indices: torch.Tensor, 
                weight: torch.Tensor | None,
                exp_avgs: torch.Tensor | None,
                exp_avg_sqs: torch.Tensor | None,
                config,
                ) -> torch.Tensor:
        '''
        scores: (..., N)
        indices: (..., N)
        weight: (NE, C)
        output: (..., C)
        output = (weight[indices, :] * scores.unsqueeze(-1)).sum(dim=-2)
        '''
        ctx.config = config
        ctx.scores_shape = scores.shape
        batch_shape = indices.shape[:-1]

        if config.ddp:
            scores_list = [torch.empty_like(scores) for _ in range(config.world_size)] if config.is_main else None
            indices_list = [torch.empty_like(indices) for _ in range(config.world_size)] if config.is_main else None
            dist.gather(scores.contiguous(), scores_list, dst=config.src_rank)
            dist.gather(indices.contiguous(), indices_list, dst=config.src_rank)
            if config.is_main:
                scores = torch.stack(scores_list, dim=0)
                indices = torch.stack(indices_list, dim=0)
                embeds = SparseEmbeddingBagFunc.forward_single(ctx, scores, indices, weight, exp_avgs, exp_avg_sqs, config)
                scatter_list = [embeds[_] for _ in range(embeds.shape[0])]
            else:
                scatter_list = None
            embed = torch.empty(batch_shape+(config.embedding_dim,), dtype=config.dtype, device=scores.device)
            dist.scatter(embed, scatter_list, src=config.src_rank)
        else:
            embed = SparseEmbeddingBagFunc.forward_single(ctx, scores, indices, weight, exp_avgs, exp_avg_sqs, config)

        return embed


    @staticmethod
    @torch.no_grad()
    def forward_single(ctx, 
                scores: torch.Tensor, 
                indices: torch.Tensor, 
                weight: torch.Tensor,
                exp_avgs: torch.Tensor,
                exp_avg_sqs: torch.Tensor,
                config,
                ) -> torch.Tensor:
        '''
        scores: (..., N)
        indices: (..., N)
        weight: (NE, C)
        output: (..., C)
        output = (weight[indices, :] * scores.unsqueeze(-1)).sum(dim=-2)
        '''
        unique_indices, inverse = torch.unique(indices.view(-1), sorted=True, return_inverse=True)
        unique_indices = unique_indices.to(torch.int32)
        ctx.save_for_backward(scores, weight, exp_avgs, exp_avg_sqs, unique_indices, inverse)
        embeds = torch.empty(unique_indices.shape + (weight.shape[-1],), dtype=config.dtype, device=unique_indices.device)
        index_to_kernel.index_to_cuda(weight, unique_indices, embeds)
        embeds = embeds[inverse].view(indices.shape + (weight.shape[-1],))
        # output = (embeds * scores.unsqueeze(-1)).sum(dim=-2)
        output = torch.einsum('...nc,...n->...c', embeds, scores) # TODO 哪个更快？
        return output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor):
        '''
        grad_output: (..., C)
        '''
        config = ctx.config
        if config.ddp:
            grad_output_list = [torch.empty_like(grad_output) for _ in range(config.world_size)] if config.is_main else None
            dist.gather(grad_output.contiguous(), grad_output_list, dst=config.src_rank)
            if config.is_main:
                grad_output = torch.stack(grad_output_list, dim=0)
                grad_scores = SparseEmbeddingBagFunc.backward_single(ctx, grad_output)
                scatter_list = [grad_scores[_] for _ in range(grad_scores.shape[0])]
            else:
                scatter_list = None
            grad_scores = torch.empty(ctx.scores_shape, device=grad_output.device, dtype=grad_output.dtype)
            dist.scatter(grad_scores, scatter_list, src=config.src_rank)
        else:
            grad_scores = SparseEmbeddingBagFunc.backward_single(ctx, grad_output)
        
        return grad_scores, None, None, None, None, None


    @staticmethod
    @torch.no_grad()
    def backward_single(ctx, grad_output: torch.Tensor):
        '''
        grad_output: (..., C)
        output:
            grad_scores: (..., N)
        '''
        C = grad_output.shape[-1]
        config = ctx.config
        scores, weight, exp_avgs, exp_avg_sqs, unique_indices, inverse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad = torch.einsum('...c,...n->...nc', grad_output, scores).view(-1, C)
        unique_grad = torch.zeros((unique_indices.shape[0], C), device=grad.device, dtype=grad.dtype)
        unique_grad.index_add_(0, inverse, grad)
        unique_grad = SparseEmbeddingBagFunc.grad_clip(unique_grad, config.grad_clip_max_norm)
        
        embeds = torch.empty(unique_indices.shape + (weight.shape[-1],), dtype=grad_output.dtype, device=unique_indices.device)
        index_to_kernel.adamw(
            weight,
            unique_grad,
            exp_avgs,
            exp_avg_sqs,
            unique_indices,
            embeds,
            config.lr,
            config.beta1,
            config.beta2,
            config.weight_decay,
            config.eps,
        )
        embeds = embeds[inverse].view(scores.shape + (weight.shape[-1],))
        grad_scores = torch.einsum('...c,...nc->...n', grad_output, embeds)
        return grad_scores

    @staticmethod
    @torch.no_grad()
    def grad_clip(grad, max_norm):
        if max_norm is None:
            return grad
        grad_norm = torch.norm(grad, p=2, dim=-1)
        scale = torch.clamp(max_norm / (grad_norm + 1e-6), max=1.0)
        grad = grad * scale.unsqueeze(-1)
        return grad
        



class SparseEmbeddingBag(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 optimizer_params: dict,
                 std: float = 0.01,
                 src_rank=0,
                 embedding_output_dtype: torch.dtype = torch.float32,
                 **kwargs,
                 ):
        super().__init__()

        if dist.is_available() and dist.is_initialized():
            local_rank = dist.get_rank()
            is_main = local_rank == src_rank
            world_size = dist.get_world_size()
            ddp = world_size > 1
        else:
            local_rank = 0
            is_main = True
            world_size = 1
            ddp = False

        if not ddp or is_main:
            self.weight = torch.nn.parameter.Parameter(torch.empty((num_embeddings, embedding_dim), device='cpu', pin_memory=True), requires_grad=False)
            nn.init.normal_(self.weight, std=std)

            self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
            self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True)
        else:
            self.weight = None
            self.exp_avgs = None
            self.exp_avg_sqs = None
        
        self.config = SimpleNamespace(
            embedding_dim=embedding_dim,
            lr=optimizer_params.get("lr", 1e-3),
            beta1=optimizer_params.get("beta1", 0.9),
            beta2=optimizer_params.get("beta2", 0.999),
            weight_decay=optimizer_params.get("weight_decay", 0.01),
            eps=optimizer_params.get("eps", 1e-8),
            grad_clip_max_norm=optimizer_params.get("grad_clip_max_norm", None),
            ddp=ddp,
            dtype=embedding_output_dtype,
            src_rank=src_rank,
            is_main=is_main,
            world_size=world_size,
        )

    def forward(self, indices: torch.Tensor, scores: torch.Tensor = None) -> torch.Tensor:
        return SparseEmbeddingBagFunc.apply(scores, indices, self.weight, self.exp_avgs, self.exp_avg_sqs, self.config)
    
    def update_lr(self, lr: float):
        self.config.lr = lr

if __name__ == "__main__":
    from accelerate import Accelerator
    accelerator = Accelerator()
    NE = 10
    C = 4
    B = 10
    N = 4
    embed = SparseEmbeddingBag(NE, C, 
        optimizer_params = {
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
    )
    if accelerator.is_main_process:
        weights_ray = embed.weight.clone().detach().cuda()
        scatter_list = [weights for _ in range(accelerator.num_processes)]
    weights = torch.empty((NE, C), device=accelerator.device, requires_grad=True)
    dist.scatter(weights, scatter_list, src=0)



    # 模拟训练步骤
    grad = torch.randn((B, C), device='cuda')

    indices = torch.randint(0, NE, (B, N), dtype=torch.int32).cuda()
    scores = torch.randn((B, N), device='cuda', requires_grad=True)
    out = embed(indices, scores)
    loss = (out * grad).sum()
    # loss.backward()
    accelerator.backward(loss)

    base_scores = scores.clone().detach().requires_grad_(True)
    base_out = weights[indices, :]
    base_out = (base_out * base_scores.unsqueeze(-1)).sum(dim=-2)

    print(torch.allclose(out, base_out))
    print((out - base_out).abs().max())

    base_loss = (base_out * grad).sum()
    accelerator.backward(base_loss)
    print(torch.allclose(scores.grad, base_scores.grad))
    print((scores.grad - base_scores.grad).abs().max())

    if accelerator.is_main_process:
        print(torch.allclose(weight_grad, weights.grad))
        print((weight_grad - weights.grad).abs().max())
        print(weights.grad)
