from torch.utils.cpp_extension import load
import os
import torch
torch.cuda.set_device(1)

pwd = os.path.dirname(os.path.abspath(__file__))

index_to_cuda_kernel = load(name='index_to_cuda', 
                            sources=[
                                os.path.join(pwd, 'index_to.cpp'), 
                                os.path.join(pwd, 'index_to_kernel.cu')
                                ], 
                            verbose=True,
                            extra_cflags=['-O3'],
                            extra_cuda_cflags=['-O3'],
                            )
def index_to_cuda(a, b):
    """
    Efficiently index into a pinned memory tensor 'a' using GPU indices 'b'
    
    Args:
        a: torch.Tensor on CPU pinned memory, shape (50000, 512), dtype float32
        b: torch.Tensor on CUDA, shape (8192,), dtype int32
        optimized: Whether to use the optimized version with deduplication (default: True)
        
    Returns:
        torch.Tensor on CUDA device with a[b, :] efficiently computed
    """
    # Validate inputs
    assert a.is_pinned(), "Tensor 'a' must be in pinned memory"
    assert b.is_cuda, "Tensor 'b' must be on CUDA device"
    assert a.dtype == torch.float32, "Tensor 'a' must be float32"
    assert b.dtype == torch.int32, "Tensor 'b' must be int32"

    return index_to_cuda_kernel.index_to_pinned(a, b)
    
if __name__ == "__main__":
    import time
    # Create input tensor in pinned memory
    a = torch.randn(50000, 512*7, dtype=torch.float32)
    a_pinned = a.pin_memory()
    
    # Create indices tensor on GPU
    b = torch.randint(0, 50000, (8192,), dtype=torch.int32, device='cuda:1')
    
    # Use our optimized kernel
    result = index_to_cuda(a_pinned, b)
    
    # Compare with naive implementation
    naive_result = a_pinned[b.cpu()].to(b.device)
    
    # Verify results
    print("Results match:", torch.allclose(result, naive_result))
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        result = index_to_cuda(a_pinned, b)
        torch.cuda.synchronize()
    end = time.time()
    print(f"Optimized method: {(end - start) / 10 * 1000:.2f} ms per call")
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        naive_result = a_pinned[b.cpu()].to(b.device, non_blocking=False)
        torch.cuda.synchronize()
    end = time.time()
    print(f"Naive method: {(end - start) / 10 * 1000:.2f} ms per call")

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        naive_result = a_pinned[b.cpu()].to(b.device, non_blocking=True)
        torch.cuda.synchronize()
    end = time.time()
    print(f"Naive method (non-blocking): {(end - start) / 10 * 1000:.2f} ms per call")
