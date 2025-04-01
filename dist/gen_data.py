import json
import random
import torch

vocab_size = 10000
category_size = 10
num_samples = 1600

proj = torch.randint(1, category_size, (vocab_size,))

data = []
for i in range(num_samples):
    index = random.randint(0, vocab_size-1)
    category = proj[index]
    data.append({'index': index, 'category': category.item()})

with open('data.json', 'w') as f:
    json.dump(data, f)
