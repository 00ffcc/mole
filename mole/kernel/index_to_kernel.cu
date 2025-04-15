#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for indexing pinned memory
__global__ void index_to_pinned_kernel(
    const float* __restrict__ input,
    const int* __restrict__ indices,
    float* __restrict__ output,
    const int batch_size,
    const int feature_dim,
    const int input_stride) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * feature_dim) {
        int batch_idx = tid / feature_dim;
        int feat_idx = tid % feature_dim;
        
        int input_idx = indices[batch_idx];
        output[tid] = input[input_idx * input_stride + feat_idx];
    }
}

torch::Tensor index_to_pinned_cuda(
    const torch::Tensor& input,
    const torch::Tensor& indices) {
    
    // Get dimensions
    int batch_size = indices.size(0);
    int feature_dim = input.size(1);
    int input_stride = input.stride(0);
    
    // Create output tensor
    auto output = torch::empty({batch_size, feature_dim}, 
                              torch::dtype(torch::kFloat32).device(indices.device()));
    
    // Calculate grid and block sizes
    const int threads_per_block = 256;
    const int blocks = (batch_size * feature_dim + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    index_to_pinned_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        indices.data_ptr<int>(),
        output.data_ptr<float>(),
        batch_size,
        feature_dim,
        input_stride
    );
    
    return output;
}