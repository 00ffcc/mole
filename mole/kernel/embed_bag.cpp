#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
void index_to_cuda(const torch::Tensor& input, const torch::Tensor& indices, const torch::Tensor& output);
void adamw(const torch::Tensor& weight, const torch::Tensor& grad, const torch::Tensor& exp_avg, const torch::Tensor& exp_avg_sq, const torch::Tensor& indices, const torch::Tensor& output_weight, const float lr, const float beta1, const float beta2, const float weight_decay, const float eps);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_to_cuda", &index_to_cuda, "pin_memory to CUDA");
    m.def("adamw", &adamw, "AdamW optimizer");
}