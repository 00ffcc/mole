#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for indexing pinned memory
// pin_memory to CUDA
template <typename T>
__global__ void index_to_cuda_kernel(
    const float* __restrict__ input,
    const int* __restrict__ indices,
    T* __restrict__ output,
    const int batch_size,
    const int feature_dim,
    const int input_stride) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (feature_dim / _k_);

    if (batch_idx < batch_size)
    {
        int feat_idx = tid % (feature_dim / _k_) * _k_;
        int output_idx = tid * _k_;
        int start_idx = indices[batch_idx] * input_stride + feat_idx;
        #pragma unroll
        for (int i=0; i<_k_; i++)
            output[output_idx + i] = T(input[start_idx + i]);
    }
}

void index_to_cuda(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& output) {
    
    // Get dimensions
    int batch_size = indices.size(0);
    int feature_dim = input.size(1);
    int input_stride = input.stride(0);
    
    
    // Calculate grid and block sizes
    const int threads_per_block = 256;
    int total_threads = batch_size * feature_dim / _k_;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    if (output.dtype() == torch::kFloat)
        index_to_cuda_kernel<<<blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int>(),
            output.data_ptr<float>(),
            batch_size,
            feature_dim,
            input_stride
        );
    else if (output.dtype() == torch::kHalf)
        index_to_cuda_kernel<<<blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int>(),
            output.data_ptr<at::Half>(),
            batch_size,
            feature_dim,
            input_stride
        );
    else if (output.dtype() == torch::kBFloat16)
        index_to_cuda_kernel<<<blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int>(),
            output.data_ptr<at::BFloat16>(),
            batch_size,
            feature_dim,
            input_stride
        );
    else
        throw std::runtime_error("Unsupported output dtype");
}
// CUDA to pin_memory
__global__ void index_to_pinned_kernel(
    float* __restrict__ input,
    const int* __restrict__ indices,
    const float* __restrict__ x,
    const int batch_size,
    const int feature_dim,
    const int input_stride) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * feature_dim) {
        int batch_idx = tid / feature_dim;
        int feat_idx = tid % feature_dim;
        
        int input_idx = indices[batch_idx];
        input[input_idx * input_stride + feat_idx] = x[tid];
    }
}
// input[indices] = x
void index_to_pinned(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& x) {
    
    // Get dimensions
    int batch_size = indices.size(0);
    int feature_dim = input.size(1);
    int input_stride = input.stride(0);
        
    // Calculate grid and block sizes
    const int threads_per_block = 256;
    const int blocks = (batch_size * feature_dim + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    index_to_pinned_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        indices.data_ptr<int>(),
        x.data_ptr<float>(),
        batch_size,
        feature_dim,
        input_stride
    );
}