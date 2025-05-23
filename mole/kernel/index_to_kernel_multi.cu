#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for indexing pinned memory
// pin_memory to CUDA
template <typename T>
__global__ void index_to_cuda_kernel(
    const float* __restrict__ input,
    const int*                indices,
          T*     __restrict__ output
    ) {
    const float* __restrict__ input_ = input + indices[blockIdx.x] * _n_ + threadIdx.x;
    T* __restrict__ output_ = output + blockIdx.x * _n_ + threadIdx.x;
    #pragma unroll
    for (int i=0; i<_n_; i+= _k_)
        output_[i] = T(input_[i]);
}

void index_to_cuda(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& output) {
    
    assert(input.size(1) == _n_);
    assert(_n_%_k_ == 0);
    
    dim3 blocksize(indices.size(0)), gridsize(_k_);
    
    // Launch kernel
    if (output.dtype() == torch::kFloat)
        index_to_cuda_kernel<<<blocksize, gridsize>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int>(),
            output.data_ptr<float>()
        );
    else if (output.dtype() == torch::kHalf)
        index_to_cuda_kernel<<<blocksize, gridsize>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int>(),
            output.data_ptr<at::Half>()
        );
    else if (output.dtype() == torch::kBFloat16)
        index_to_cuda_kernel<<<blocksize, gridsize>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int>(),
            output.data_ptr<at::BFloat16>()
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

__global__ void adamw_kernel(
    float* __restrict__ weight_,
    const float* __restrict__ grad_,
    float* __restrict__ exp_avg_,
    float* __restrict__ exp_avg_sq_,
    const int* __restrict__ indices,
    const int batch_size,
    const int feature_dim,
    const int input_stride,
    const float lr,
    const float _beta1, // 1-beta1
    const float _beta2, // 1-beta2
    const float weight_decay,
    const float eps) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * feature_dim)
    {
        int param_idx = indices[tid / feature_dim] * input_stride + tid % feature_dim;
        float grad = grad_[tid];
        float exp_avg = exp_avg_[param_idx];
        float exp_avg_sq = exp_avg_sq_[param_idx];
        float weight = weight_[param_idx];

        exp_avg += _beta1 * (grad - exp_avg);
        exp_avg_sq += _beta2 * (grad * grad - exp_avg_sq);
        weight -= lr * (weight_decay * weight + exp_avg / (sqrt(exp_avg_sq) + eps));

        weight_[param_idx] = weight;
        exp_avg_[param_idx] = exp_avg;
        exp_avg_sq_[param_idx] = exp_avg_sq;
    }
}

void adamw(
    const torch::Tensor& weight,
    const torch::Tensor& grad,
    const torch::Tensor& exp_avg,
    const torch::Tensor& exp_avg_sq,
    const torch::Tensor& indices,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps
    ) {
    
    // Get dimensions
    int batch_size = indices.size(0);
    int feature_dim = weight.size(1);
    int input_stride = weight.stride(0);
        
    // Calculate grid and block sizes
    const int threads_per_block = 256;
    const int blocks = (batch_size * feature_dim + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    adamw_kernel<<<blocks, threads_per_block>>>(
        weight.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size,
        feature_dim,
        input_stride,
        lr,
        1 - beta1,
        1 - beta2,
        weight_decay,
        eps
    );
}