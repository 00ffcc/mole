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
    
    if (tid < batch_size * feature_dim) {
        int batch_idx = tid / feature_dim;
        int feat_idx = tid % feature_dim;
        
        int input_idx = indices[batch_idx];
        output[tid] = T(input[input_idx * input_stride + feat_idx]);
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
    const int blocks = (batch_size * feature_dim + threads_per_block - 1) / threads_per_block;
    
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

template <typename T>
__global__ void adamw_kernel(
    float* __restrict__ weight_,
    const T* __restrict__ grad_,
    float* __restrict__ exp_avg_,
    float* __restrict__ exp_avg_sq_,
    const int* __restrict__ indices,
    T* __restrict__ output_weight,
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
        output_weight[tid] = weight;

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
    const torch::Tensor& output_weight,
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
    if (grad.dtype() == torch::kFloat)
        adamw_kernel<<<blocks, threads_per_block>>>(
            weight.data_ptr<float>(),
            grad.data_ptr<float>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            indices.data_ptr<int>(),
            output_weight.data_ptr<float>(),
            batch_size,
            feature_dim,
            input_stride,
            lr,
            1 - beta1,
            1 - beta2,
            weight_decay,
            eps
        );
    else if (grad.dtype() == torch::kHalf)
        adamw_kernel<<<blocks, threads_per_block>>>(
            weight.data_ptr<float>(),
            grad.data_ptr<at::Half>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            indices.data_ptr<int>(),
            output_weight.data_ptr<at::Half>(),
            batch_size,
            feature_dim,
            input_stride,
            lr,
            1 - beta1,
            1 - beta2,
            weight_decay,
            eps
        );
    else if (grad.dtype() == torch::kBFloat16)
        adamw_kernel<<<blocks, threads_per_block>>>(
            weight.data_ptr<float>(),
            grad.data_ptr<at::BFloat16>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            indices.data_ptr<int>(),
            output_weight.data_ptr<at::BFloat16>(),
            batch_size,
            feature_dim,
            input_stride,
            lr,
            1 - beta1,
            1 - beta2,
            weight_decay,
            eps
        );
    else
        throw std::runtime_error("Unsupported grad dtype");
}