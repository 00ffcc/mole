from accelerate import Accelerator, ProfileKwargs
import torch.distributed as dist
import os
import torch

# world_size = int(os.environ.get("WORLD_SIZE", 2)) # Default to 2 if not set
# rank = int(os.environ.get("RANK", -1))           # torchrun sets these
# dist.init_process_group("gloo", rank=rank, world_size=world_size)

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="./log/scatter"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

tensor_shape = (2048, 16000)
tensor_dtype = torch.float32

with accelerator.profile() as prof:
    for _ in range(10):
        if accelerator.is_local_main_process:
            scatter_list = [torch.randint(0, 10000, tensor_shape, dtype=tensor_dtype, device='cpu') for _ in range(2)]
            scatter_list = [t.to(accelerator.device) for t in scatter_list]
        else:
            scatter_list = None

        out = torch.empty(tensor_shape, device=accelerator.device)
        dist.scatter(out, scatter_list, src=0)

        dist.barrier()
        prof.step()
