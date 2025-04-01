from accelerate import Accelerator, ProfileKwargs
from matplotlib import category
import torch.distributed as dist
import torch
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
        self.embedding = torch.nn.Embedding(vocab_size, dim_embed)
        self.fc = torch.nn.Linear(dim_embed, n)
    def forward(self, x):
        return self.fc(self.embedding(x))

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


is_profile = False
with accelerator.profile() if is_profile else contextlib.suppress() as prof:
    for epoch in range(10):
        losss = []
        for i, (index, category) in enumerate(dataloader):
            out = model(index)
            loss = criterion(out, category)
            losss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # dist.barrier()
            # prof.step()
        print(f"epoch {epoch}, loss {sum(losss)/len(losss)}")
