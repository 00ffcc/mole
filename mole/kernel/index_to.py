from torch.utils.cpp_extension import load
import os
import torch

pwd = os.path.dirname(os.path.abspath(__file__))

index_to_kernel = load(name='index_to_cuda', 
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
        a: torch.Tensor on CPU pinned memory, dtype float32
        b: torch.Tensor on CUDA
        
    Returns:
        torch.Tensor on CUDA device with a[b, :] efficiently computed
    """
    # Validate inputs
    assert a.is_pinned(), "Tensor 'a' must be in pinned memory"
    assert b.is_cuda, "Tensor 'b' must be on CUDA device"
    assert a.dtype == torch.float32, "Tensor 'a' must be float32"
    assert b.dtype == torch.int32, "Tensor 'b' must be int32"

    return index_to_kernel.index_to_cuda(a, b)

def index_to_pinned(a, b, c):
    """
    a[b] = c
    a: torch.Tensor on CPU pinned memory, dtype float32
    b: torch.Tensor on CUDA
    c: torch.Tensor on CUDA
    
    Returns:
        None
    """
    # Validate inputs
    assert a.is_pinned(), "Tensor 'a' must be in pinned memory"
    assert b.is_cuda, "Tensor 'b' must be on CUDA device"
    assert c.is_cuda, "Tensor 'c' must be on CUDA device"
    assert a.dtype == torch.float32, "Tensor 'a' must be float32"
    assert b.dtype == torch.int32, "Tensor 'b' must be int32"
    assert c.dtype == torch.float32, "Tensor 'c' must be float32"

    index_to_kernel.index_to_pinned(a, b, c)

    
if __name__ == "__main__":
    import time
    torch.cuda.set_device(1)
    epoch = 40
    if True:
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
        for _ in range(epoch):
            result = index_to_cuda(a_pinned, b)
            torch.cuda.synchronize()
        end = time.time()
        print(f"Optimized method: {(end - start) / epoch * 1000:.2f} ms per call")
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(epoch):
            naive_result = a_pinned[b.cpu()].to(b.device, non_blocking=False)
            torch.cuda.synchronize()
        end = time.time()
        print(f"Naive method: {(end - start) / epoch * 1000:.2f} ms per call")

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(epoch):
            naive_result = a_pinned[b.cpu()].to(b.device, non_blocking=True)
            torch.cuda.synchronize()
        end = time.time()
        print(f"Naive method (non-blocking): {(end - start) / epoch * 1000:.2f} ms per call")
    if False:
        # Create input tensor in pinned memory
        a = torch.randn(50000, 512*7, dtype=torch.float32, pin_memory=True)
        b = torch.randint(0, 50000, (8192,), dtype=torch.int32, device='cuda:1')
        c = torch.randn(8192, 512*7, dtype=torch.float32, device='cuda:1')

        res1 = a.clone()
        index_to_pinned(a, b, c)
        res1[b.cpu()] = c.cpu()

        print("Results match:", torch.allclose(a, res1))

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(epoch):
            index_to_pinned(a, b, c)
            torch.cuda.synchronize()
        end = time.time()
        print(f"Optimized method: {(end - start) / epoch * 1000:.2f} ms per call")
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(epoch):
            res1[b.cpu()] = c.cpu()
            torch.cuda.synchronize()
        end = time.time()
        print(f"Naive method: {(end - start) / epoch * 1000:.2f} ms per call")


