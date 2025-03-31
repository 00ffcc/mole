import torch
import torch.distributed as dist
import os
import time
from accelerate import Accelerator, ProfileKwargs

world_size = int(os.environ.get("WORLD_SIZE", 2)) # Default to 2 if not set
rank = int(os.environ.get("RANK", -1))           # torchrun sets these
dist.init_process_group("gloo", rank=rank, world_size=world_size)

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="log/send"
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

def run_send_recv(rank):

    # --- Define source and destination ranks ---
    src_rank = 0
    dst_rank = 1

    tensor_shape = (2048, 16000)
    tensor_dtype = torch.float32

    if rank == src_rank:
        # --- Source Process (Rank 0) ---
        tensor_to_send = torch.randint(0, 10000, tensor_shape, dtype=tensor_dtype)
        # tensor_to_send.share_memory_()
        print(f"Rank {rank} (src): Preparing to send tensor")
        print(f"Rank {rank} (src): Tensor device: {tensor_to_send.device}")

        # Send the tensor to the destination rank
        dist.send(tensor=tensor_to_send, dst=dst_rank)
        print(f"Rank {rank} (src): Tensor sent to rank {dst_rank}.")

    elif rank == dst_rank:
        # --- Destination Process (Rank 1) ---
        # Create an empty tensor buffer on the CPU with the expected shape and dtype
        received_tensor = torch.empty(tensor_shape, dtype=tensor_dtype)
        print(f"Rank {rank} (dst): Preparing to receive tensor (shape: {tensor_shape}, dtype: {tensor_dtype}).")
        print(f"Rank {rank} (dst): Buffer device: {received_tensor.device}")

        # Receive the tensor from the source rank. This blocks until data arrives.
        dist.recv(tensor=received_tensor, src=src_rank)
        received_tensor.to(accelerator.device)
        print(f"Rank {rank} (dst): Tensor received from rank {src_rank}")
        print(f"Rank {rank} (dst): Received tensor device: {received_tensor.device}")

    else:
        # Other ranks don't participate in this specific send/recv
        print(f"Rank {rank}: Not participating in this send/recv.")
        pass

    # Add a barrier to ensure all processes reach this point before cleanup
    print(f"Rank {rank}: Reached barrier.")
    dist.barrier()
    # time.sleep(0.1) # Short sleep to allow prints to flush before cleanup potentially clears them

    print(f"Rank {rank}: Execution finished.")


if __name__ == "__main__":
    # Use torchrun or manually set environment variables RANK and WORLD_SIZE
    # Example for manual execution (run this script in separate terminals):
    # Terminal 1: RANK=0 WORLD_SIZE=2 python your_script.py
    # Terminal 2: RANK=1 WORLD_SIZE=2 python your_script.py

    # Better: Use torchrun (recommended)
    # torchrun --nproc_per_node=2 your_script.py

    rank = accelerator.local_process_index

    with accelerator.profile() as prof:
        for _ in range(10):
            run_send_recv(rank)
            prof.step()