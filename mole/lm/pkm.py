import torch
from torch import nn
from mole.dist_embed.embed import SparseEmbedding
from mole.dist_embed.embed_bag import SparseEmbeddingBag

class PKM(nn.Module):

    def __init__(self, config):

        super().__init__()

        # global parameters
        self.input_dim = config.dim
        self.output_dim = config.dim
        self.k_dim = config.pkm_k_dim
        self.v_dim = config.dim
        self.n_keys = config.pkm_n_keys
        self.size = self.n_keys ** 2
        self.heads = config.pkm_heads
        self.knn = config.pkm_knn # 取knn个
        assert self.k_dim >= 2 and self.k_dim % 2 == 0
        
        # 避免注册为子模块
        self.values = [
                        SparseEmbeddingBag(
                           num_embeddings         = self.size, 
                           embedding_dim          = self.v_dim, 
                           optimizer_params       = config.optimizer_params, 
                           std                    = self.v_dim ** -0.5, 
                           embedding_output_dtype = config.embedding_output_dtype
                       )] if config.use_embedding_bag else [
                        SparseEmbedding(
                           num_embeddings         = self.size, 
                           embedding_dim          = self.v_dim, 
                           optimizer_params       = config.optimizer_params, 
                           std                    = self.v_dim ** -0.5, 
                           embedding_output_dtype = config.embedding_output_dtype
                       )]

        # query network
        self.query_proj = nn.Linear(self.input_dim, self.heads * self.k_dim, bias=True)

        self.keys = nn.Parameter(torch.empty(self.heads, 2, self.n_keys, self.k_dim // 2))
        bound = (self.k_dim // 2) ** -0.5
        nn.init.uniform_(self.keys, -bound, bound)

        # self.ln_q = nn.LayerNorm(self.k_dim)
        # nn.init.constant_(self.ln_q.weight, 1 / 3000)
        # self.ln_k = nn.LayerNorm(self.k_dim // 2)



    def get_indices(self, query):
        # query = self.ln_q(query)
        # key = self.ln_k(self.keys)
        key = self.keys
        query = query.view(-1, self.heads, 2, self.k_dim // 2, 1)
        scores = (key @ query).view(-1, self.heads, 2, self.n_keys) # (B, H, 2, n_keys)

        scores1 = scores[:, :, 0, :]
        scores2 = scores[:, :, 1, :]
        
        scores1, indices1 = scores1.topk(self.knn, dim=-1)
        scores2, indices2 = scores2.topk(self.knn, dim=-1)

        all_scores = (
            scores1.view(-1, self.heads, self.knn, 1).expand(-1, -1, -1, self.knn) +
            scores2.view(-1, self.heads, 1, self.knn).expand(-1, -1, self.knn, -1)
        ).view(-1, self.heads, self.knn ** 2)
        
        all_indices = (
            indices1.view(-1, self.heads, self.knn, 1).expand(-1, -1, -1, self.knn) * self.n_keys +
            indices2.view(-1, self.heads, 1, self.knn).expand(-1, -1, self.knn, -1)
        ).view(-1, self.heads, self.knn ** 2)

        scores, best_indices = torch.topk(all_scores, k=self.knn, dim=-1)
        indices = all_indices.gather(-1, best_indices)
        return scores, indices


    def forward(self, x):
        prefix_shape = x.shape[:-1]

        # compute query
        query = self.query_proj(x).view(-1, self.heads, self.k_dim) 


        # retrieve indices and scores
        scores, indices = self.get_indices(query)

        # merge heads / knn (since we sum heads)
        indices = indices.view(-1, self.heads * self.knn)
        scores = scores.float().softmax(dim=-1).type_as(scores)
        scores = scores.view(-1, self.heads * self.knn)

        # weighted sum of values
        if isinstance(self.values[0], SparseEmbedding):
            output = self.values[0](indices)
            output = (output * scores.unsqueeze(-1)).sum(dim=-2)
        else:
            output = self.values[0](indices, scores)

        return output.view(prefix_shape + (self.v_dim,))
    
if __name__ == '__main__':
    from types import SimpleNamespace
    config = SimpleNamespace(
        dim=10,
        pkm_k_dim=16,
        pkm_n_keys=4,
        pkm_heads=2,
        pkm_knn=2,
    )
    values = SparseEmbedding(16, 10, 
                    {
                        "lr": 1e-3,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "weight_decay": 0.01,
                        "eps": 1e-8,
                    }, 
                    std=0.01)
    device = 'cuda:0'
    model = PKM(config, values).to(device)
    for name, param in model.named_parameters():
        print(name, param.shape, param.device)
    x = torch.randn(4, 10).to(device)
    y = model(x)
    print(y.shape)