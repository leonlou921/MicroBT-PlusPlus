#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

#ifndef MICROBT_PF_CUDA_BUILD_ID
#define MICROBT_PF_CUDA_BUILD_ID "missing"
#endif

void microbt_pf_cuda_device_check() {
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
}

const char* microbt_pf_cuda_build_id() {
    return MICROBT_PF_CUDA_BUILD_ID;
}

namespace {

__device__ __forceinline__ int32_t find_root(int32_t* parent, int32_t vertex) {
    int32_t root = vertex;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[vertex] != vertex) {
        const int32_t next = parent[vertex];
        parent[vertex] = root;
        vertex = next;
    }
    return root;
}

__device__ __forceinline__ int32_t find_root_readonly(
    const int32_t* parent, int32_t vertex) {
    while (parent[vertex] != vertex) {
        vertex = parent[vertex];
    }
    return vertex;
}

__device__ __forceinline__ int32_t link_elder(
    int32_t* parent,
    const float* values,
    int32_t root_a,
    int32_t root_b) {
    const bool a_is_elder =
        values[root_a] > values[root_b] ||
        (values[root_a] == values[root_b] && root_a < root_b);
    const int32_t survivor = a_is_elder ? root_a : root_b;
    const int32_t killed = a_is_elder ? root_b : root_a;
    parent[killed] = survivor;
    return killed;
}

__global__ void persistence_pairs_kernel(
    const float* values,
    const int64_t* order,
    int32_t depth,
    int32_t height,
    int32_t width,
    int32_t size,
    int32_t* parent,
    uint8_t* active,
    int64_t* birth_out,
    int64_t* death_out,
    uint8_t* essential_out,
    int32_t* count_out) {
    if (blockIdx.x != 0 || threadIdx.x >= 32) {
        return;
    }
    __shared__ int32_t neighbor_roots[26];
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    int32_t pair_count = 0;
    const int32_t plane = height * width;
    for (int32_t position = 0; position < size; ++position) {
        const int32_t vertex = static_cast<int32_t>(order[position]);
        if (lane == 0) {
            parent[vertex] = vertex;
            active[vertex] = uint8_t{1};
        }
        __syncwarp();
        const int32_t z = vertex / plane;
        const int32_t remainder = vertex - z * plane;
        const int32_t y = remainder / width;
        const int32_t x = remainder - y * width;
        if (lane < 26) {
            const int32_t raw_offset = lane < 13 ? lane : lane + 1;
            const int32_t dz = raw_offset / 9 - 1;
            const int32_t dy = (raw_offset % 9) / 3 - 1;
            const int32_t dx = raw_offset % 3 - 1;
            const int32_t nz = z + dz;
            const int32_t ny = y + dy;
            const int32_t nx = x + dx;
            if (nz < 0 || ny < 0 || nx < 0 ||
                nz >= depth || ny >= height || nx >= width) {
                neighbor_roots[lane] = -1;
            } else {
                const int32_t neighbor = (nz * height + ny) * width + nx;
                neighbor_roots[lane] = active[neighbor]
                    ? find_root_readonly(parent, neighbor)
                    : -1;
            }
        }
        __syncwarp();
        if (lane == 0) {
            for (int32_t neighbor_position = 0; neighbor_position < 26;
                 ++neighbor_position) {
                    const int32_t neighbor_root = neighbor_roots[neighbor_position];
                    if (neighbor_root < 0) {
                        continue;
                    }
                    const int32_t root_vertex = find_root(parent, vertex);
                    const int32_t root_neighbor = find_root(parent, neighbor_root);
                    if (root_vertex == root_neighbor) {
                        continue;
                    }
                    const int32_t killed =
                        link_elder(parent, values, root_vertex, root_neighbor);
                    if (values[killed] > values[vertex]) {
                        birth_out[pair_count] = static_cast<int64_t>(killed);
                        death_out[pair_count] = static_cast<int64_t>(vertex);
                        essential_out[pair_count] = uint8_t{0};
                        ++pair_count;
                    }
            }
        }
        __syncwarp();
    }
    if (lane == 0) {
        const int32_t essential_owner = find_root(parent, 0);
        birth_out[pair_count] = static_cast<int64_t>(essential_owner);
        death_out[pair_count] = int64_t{-1};
        essential_out[pair_count] = uint8_t{1};
        *count_out = pair_count + 1;
    }
}

}  // namespace

std::vector<torch::Tensor> microbt_pf_persistence_pairs_cuda(
    torch::Tensor field, torch::Tensor order) {
    TORCH_CHECK(field.is_cuda(), "field must be a CUDA tensor");
    TORCH_CHECK(field.scalar_type() == torch::kFloat32, "field must be float32");
    TORCH_CHECK(field.dim() == 3 && field.numel() > 0, "field must be nonempty 3D");
    TORCH_CHECK(field.is_contiguous(), "field must be contiguous");
    TORCH_CHECK(order.is_cuda(), "order must be a CUDA tensor");
    TORCH_CHECK(order.scalar_type() == torch::kInt64, "order must be int64");
    TORCH_CHECK(order.is_contiguous(), "order must be contiguous");
    TORCH_CHECK(order.numel() == field.numel(), "order size must equal field size");
    TORCH_CHECK(field.numel() <= INT32_MAX, "field is too large for exact CUDA H0");

    c10::cuda::CUDAGuard device_guard(field.device());
    const int32_t size = static_cast<int32_t>(field.numel());
    const auto int64_options = field.options().dtype(torch::kInt64);
    const auto int32_options = field.options().dtype(torch::kInt32);
    const auto byte_options = field.options().dtype(torch::kUInt8);
    auto parent = torch::empty({size}, int32_options);
    auto active = torch::zeros({size}, byte_options);
    auto birth = torch::empty({size}, int64_options);
    auto death = torch::empty({size}, int64_options);
    auto essential = torch::empty({size}, byte_options);
    auto count = torch::zeros({1}, int32_options);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(field.get_device());
    persistence_pairs_kernel<<<1, 32, 0, stream>>>(
        field.data_ptr<float>(),
        order.data_ptr<int64_t>(),
        static_cast<int32_t>(field.size(0)),
        static_cast<int32_t>(field.size(1)),
        static_cast<int32_t>(field.size(2)),
        size,
        parent.data_ptr<int32_t>(),
        active.data_ptr<uint8_t>(),
        birth.data_ptr<int64_t>(),
        death.data_ptr<int64_t>(),
        essential.data_ptr<uint8_t>(),
        count.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {birth, death, essential, count};
}
