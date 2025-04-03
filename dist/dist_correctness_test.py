from accelerate import Accelerator, ProfileKwargs
from matplotlib import category
import torch.distributed as dist
import torch
from embedding_offload.embedding_adamw import SparseEmbedding
import contextlib

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log/dist_training"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

vocab_size = 10000
dim_embed = 300
B = 8
category_size = 100
class classifier(torch.nn.Module):
    def __init__(self, n, dim_embed):
        super().__init__()
        self.fc = torch.nn.Linear(dim_embed, n)
    def forward(self, embed):
        return self.fc(embed)

model = classifier(category_size, dim_embed)

from torch.utils.data import DataLoader, Dataset
class MyDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            import json
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]['index'], self.data[index]['category']

dataset = MyDataset('data.json')
dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()

dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

if accelerator.is_local_main_process:
    embedding = SparseEmbedding(vocab_size, dim_embed, optimizer_params = {
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0001,
                "eps": 1e-8,
            })
is_profile = False
with accelerator.profile() if is_profile else contextlib.suppress() as prof:
    for epoch in range(100):
        losss = []
        for i, (index, category) in enumerate(dataloader):

            gather_list = [torch.empty_like(index) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
            dist.gather(index, gather_list, dst=0)
            if accelerator.is_local_main_process:
                indexs = torch.stack(gather_list, dim=0)
                embeds = embedding(indexs)
                scatter_list = [embeds[_] for _ in range(embeds.shape[0])]
            else:
                scatter_list = None

            embed = torch.empty(index.shape + (dim_embed,), device=accelerator.device)
            dist.scatter(embed, scatter_list, src=0)
            if accelerator.is_local_main_process:
                del embeds, indexs, gather_list, scatter_list
   
            embed.requires_grad_(True)
            embed.retain_grad()

            out = model(embed)
            loss = criterion(out, category)
            losss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gather_list = [torch.empty_like(embed.grad) for _ in range(dist.get_world_size())] if accelerator.is_local_main_process else None
            dist.gather(embed.grad, gather_list, dst=0)
            if accelerator.is_local_main_process:
                grad = torch.cat(gather_list, dim=0)
                embedding.apply_gradients(grad)
            # dist.barrier()
            # prof.step()
        print(f"epoch {epoch}, loss {sum(losss)/len(losss)}")
