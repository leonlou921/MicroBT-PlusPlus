#include <torch/extension.h>

#include <string>
#include <vector>

void microbt_pf_cuda_device_check();
const char* microbt_pf_cuda_build_id();
std::vector<torch::Tensor> microbt_pf_persistence_pairs_cuda(
    torch::Tensor field, torch::Tensor order);

std::string build_id() {
    microbt_pf_cuda_device_check();
    return microbt_pf_cuda_build_id();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("build_id", &build_id);
    module.def(
        "persistence_pairs",
        &microbt_pf_persistence_pairs_cuda,
        pybind11::arg("field"),
        pybind11::arg("order"));
}
