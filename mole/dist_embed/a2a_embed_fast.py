"""
number-wise
在all_to_all之前进行去重，减少all_to_all传递冗余embedding的开销
token可以全部都传
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import torch.distributed as dist
import os
from types import SimpleNamespace
import math

str2dtype = {
            "float32" : torch.float32,
            "bfloat16": torch.bfloat16,
            "fp32"    : torch.float32,
            "bf16"    : torch.bfloat16,
        }

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
@torch.no_grad()
def grad_clip(grad, max_norm):
    if max_norm is None:
        return grad
    grad_norm = torch.norm(grad, p=2, dim=-1)
    scale = torch.clamp(max_norm / (grad_norm + 1e-6), max=1.0)
    grad = grad * scale.unsqueeze(-1)
    return grad

class SparseEmbeddingLayer:
    def __init__(
            self,
            num_embeddings: int, 
            embedding_dim: int, 
            optimizer_params: dict,
            embedding_output_dtype: torch.dtype | str = torch.float32,
            params_dtype: torch.dtype | str = torch.float32,
    ):
        self.embedding_output_dtype = str2dtype[embedding_output_dtype.lower()] if isinstance(embedding_output_dtype, str) else embedding_output_dtype
        params_dtype = str2dtype[params_dtype.lower()] if isinstance(params_dtype, str) else params_dtype

        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        local_emb_num = (num_embeddings - 1 - local_rank) // world_size + 1
        self.weight = torch.empty((local_emb_num, embedding_dim), device='cpu', pin_memory=True, requires_grad=True, dtype=params_dtype)
        self.exp_avgs = torch.zeros_like(self.weight, device='cpu', pin_memory=True, dtype=params_dtype)
        self.exp_avg_sqs = torch.zeros_like(self.weight, device='cpu', pin_memory=True, dtype=params_dtype)

        self.lr=optimizer_params.get("lr", 1e-3)
        self.beta1=optimizer_params.get("beta1", 0.9)
        self.beta2=optimizer_params.get("beta2", 0.999)
        self.weight_decay=optimizer_params.get("weight_decay", 0.001)
        self.eps=optimizer_params.get("eps", 1e-8)
        self.grad_clip_max_norm=optimizer_params.get("grad_clip_max_norm", None)

        self.stream = torch.cuda.Stream()
    
    @torch.no_grad()
    def forward_1(self):
        '''
        all_to_all之前
        '''
        print(f"global_unique_indices:{self.global_unique_indices}")
        embeds = torch.empty(
            self.global_unique_indices.shape + (self.weight.shape[-1],), 
            dtype=self.embedding_output_dtype, 
            device=self.global_unique_indices.device,
        )
        index_to_kernel.index_to_cuda(
            self.weight,
            self.global_unique_indices, 
            embeds,
        )
        global_unique_embeds = embeds[self.global_inverse]
        self.local_unique_embeds = torch.empty(
            (self.local_unique_size, self.weight.shape[-1]),
            dtype=self.embedding_output_dtype,
            device=self.global_unique_indices.device,
        )
        print(f"global_unique_embeds:{global_unique_embeds}")
        self.handle = dist.all_to_all_single(
            self.local_unique_embeds,
            global_unique_embeds,
            output_split_sizes=self.output_length_per_rank,
            input_split_sizes=self.length_per_rank,
            async_op=True,
        )

    @torch.no_grad()
    def forward_2(self):
        '''
        all_to_all之后
        '''
        self.handle.wait()
        self.handle = None
        print(f"local_unique_embeds:{self.local_unique_embeds}")
        embeds = self.local_unique_embeds[self.local_inverse].view(*self.local_indices_shape, -1)

        return embeds

    @torch.no_grad()
    def backward_1(self, grad_output):
        '''
        all_to_all之前
        '''
        C = self.weight.shape[-1]
        unique_grads = torch.zeros(
            (self.local_unique_size, C),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        print(f"local_inverse:{self.local_inverse}")
        unique_grads.index_add_(0, self.local_inverse, grad_output.view(-1, C)) #
        print(f"unique_grads:{unique_grads}")
        self.global_grads = torch.empty(
            (self.global_inverse.shape[0], C),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        self.handle = dist.all_to_all_single(
            self.global_grads,
            unique_grads,
            output_split_sizes=self.length_per_rank,
            input_split_sizes=self.output_length_per_rank,
            async_op=True,
        )

    @torch.no_grad()
    def backward_2(self):
        '''
        all_to_all之后
        '''
        self.handle.wait()
        self.handle = None
        global_unique_grads = torch.zeros(
            (self.global_unique_indices.shape[0], self.weight.shape[-1]),
            dtype=self.global_grads.dtype,
            device=self.global_grads.device,
        )
        global_unique_grads.index_add_(0, self.global_inverse, self.global_grads)
        print(f"global_unique_grads:{global_unique_grads}")

        global_unique_grads = grad_clip(global_unique_grads, self.grad_clip_max_norm)

        index_to_kernel.adamw(
            self.weight,
            global_unique_grads,
            self.exp_avgs,
            self.exp_avg_sqs,
            self.global_unique_indices,
            self.lr,
            self.beta1,
            self.beta2,
            self.weight_decay,
            self.eps,
        )



class SparseEmbedding:
    
    def __init__(
            self,
            num_layers: int,
            **kwargs,
    ):
        assert dist.is_available() and dist.is_initialized()
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_embeddings = kwargs['num_embeddings']

        self.layers = [
            SparseEmbeddingLayer(
                **kwargs,
            )
            for _ in range(num_layers)
        ]

    @torch.no_grad()
    def dispatch(self, local_indices: torch.Tensor):
        '''
        indices: [B, N]
        global: 当前rank负责的所有rank的信息
        local: 当前rank需要输出的信息
        '''
        local_indices_shape = local_indices.shape
        local_indices = local_indices.to(torch.int32).view(-1)
        indices_list = [torch.empty_like(local_indices) for _ in range(self.world_size)]
        dist.all_gather(indices_list, local_indices.contiguous())

        # 如何scatter
        global_indices = torch.stack(indices_list, dim=0) # [W, B*N]
        mask = global_indices % self.world_size == self.local_rank
        unique_indices_per_rank = [
            global_indices[i][mask[i]].unique(sorted=True)
            for i in range(self.world_size)
        ]
        global_unique_indices, global_inverse = torch.cat(unique_indices_per_rank, dim=0).unique(sorted=True, return_inverse=True)
        length_per_rank = [unique_indices_per_rank[i].shape[0] for i in range(self.world_size)]

        # 如何gather
        local_indices_per_rank = [
            local_indices[local_indices % self.world_size == i].unique(sorted=True)
            for i in range(self.world_size)
        ]
        output_length_per_rank = [local_indices_per_rank[i].shape[0] for i in range(self.world_size)]
        local_indices_per_rank = torch.concat(local_indices_per_rank, dim=0)
        print("local_indices_per_rank", local_indices_per_rank)
        mapping = torch.zeros(self.num_embeddings, dtype=torch.int32, device=local_indices.device)
        mapping[local_indices_per_rank] = torch.arange(local_indices_per_rank.shape[0], device=local_indices.device, dtype=torch.int32)
        local_inverse = mapping[local_indices]
    
        for layer in self.layers:
            layer.global_unique_indices = global_unique_indices // self.world_size # w_local[i] = w_global[i*world_size+local_rank]
            layer.global_inverse = global_inverse
            layer.length_per_rank = length_per_rank
            layer.local_indices_shape = local_indices_shape
            layer.local_inverse = local_inverse
            layer.local_unique_size = local_indices_per_rank.shape[0]
            layer.output_length_per_rank = output_length_per_rank


    @torch.no_grad()
    def combine(self, layer_idx: int):
        '''
        按顺序调用
        '''
        if layer_idx == 0:
            with torch.cuda.stream(self.layers[layer_idx].stream):
                self.layers[layer_idx].forward_1()

        with torch.cuda.stream(self.layers[layer_idx].stream):
            embeds = self.layers[layer_idx].forward_2()
        
        if layer_idx < len(self.layers) - 1:
            with torch.cuda.stream(self.layers[layer_idx+1].stream):
                self.layers[layer_idx+1].forward_1()

        def backward_hook(grad):
            with torch.cuda.stream(self.layers[layer_idx].stream):
                self.layers[layer_idx].backward_1(grad)
            if layer_idx < len(self.layers) - 1:
                with torch.cuda.stream(self.layers[layer_idx+1].stream):
                    self.layers[layer_idx+1].backward_2()
            if layer_idx == 0:
                with torch.cuda.stream(self.layers[layer_idx].stream):
                    self.layers[layer_idx].backward_2()
        
        embeds.requires_grad_(True)
        embeds.register_hook(backward_hook)

        return embeds
    
    @torch.no_grad()
    def update_lr(self, lr: float)
        for layer in self.layers:
            layer.lr = lr

if __name__ == "__main__":
    from accelerate import Accelerator
    accelerator = Accelerator()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    NE = 2
    C = 3
    B = 1
    N = 2

    sparse_embedding = SparseEmbedding(
        num_layers=2,
        num_embeddings=NE,
        embedding_dim=C,
        optimizer_params={
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.001,
            "eps": 1e-8,
            "grad_clip_max_norm": 1.0,
        },
        embedding_output_dtype=torch.float32,
        params_dtype=torch.float32,
    )
    # 重复c次
    sparse_embedding.layers[0].weight = torch.arange(start=local_rank, end=NE, step=world_size, device=local_rank, dtype=torch.float32).repeat(C).view(-1, C)

    sparse_embedding.dispatch(torch.tensor([[0, 1], [1, 1]], device=local_rank, dtype=torch.int32))

    res0 = sparse_embedding.combine(0)
    print(res0)
    res1 = sparse_embedding.combine(1)

    res1.backward(torch.ones_like(res1))
    res0.backward(torch.ones_like(res0))
