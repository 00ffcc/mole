#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
void index_to_cuda(const torch::Tensor& input, const torch::Tensor& indices, const torch::Tensor& output);
void index_to_pinned(const torch::Tensor& input, const torch::Tensor& indices, const torch::Tensor& x);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_to_cuda", &index_to_cuda, "pin_memory to CUDA");
    m.def("index_to_pinned", &index_to_pinned, "CUDA to pin_memory");
}