/**
 * @file sum_device.cu
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-10
 *
 * @copyright Copyright (c) 2019
 */
#include "math/sum.h"

#define BLK_DIM 1024
#define BLK2D_DIM 32

namespace magmadnn {
namespace math {

template <typename T>
__global__ void kernel_sum_device(T** tensors, T* out, size_t size, unsigned int n_tensors) {
    index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    index_t stride = gridDim.x * blockDim.x;

    for (index_t i = idx; i < size; i += stride) {
        out[i] = tensors[0][i];
    }

    /* x_idx will span the individual tensor sizes */
    for (unsigned int t = 1; t < n_tensors; t++) {
        for (size_t i = idx; i < size; i += stride) {
            out[i] += tensors[t][i];
        }
    }
}

template <typename T>
void sum_device(const std::vector<std::reference_wrapper<const Tensor>>& tensors, Tensor &out) {
    size_t size = out.size();

    T** tensors_arr;
    const T ** host_tensors_arr;
    host_tensors_arr = (const T**) malloc(tensors.size() * sizeof(T*));
    cudaErrchk(cudaMalloc((void**) &tensors_arr, tensors.size() * sizeof(T*)));
    
    for (size_t i = 0; i < tensors.size(); i++) {
        host_tensors_arr[i] = tensors[i].get().get_ptr<T>();
    }
    
    cudaErrchk(cudaMemcpy(tensors_arr, host_tensors_arr, tensors.size() * sizeof(T*), cudaMemcpyHostToDevice));

    kernel_sum_device<<<(size+BLK_DIM-1)/BLK_DIM,BLK_DIM>>> (tensors_arr, out.get_ptr<T>(), size, tensors.size());

    cudaErrchk(cudaFree(tensors_arr));
    free(host_tensors_arr);
}
#define COMPILE_SUM_DEVICE(type) template void sum_device<type>(const std::vector<std::reference_wrapper<const Tensor>>&, Tensor&);
CALL_FOR_ALL_TYPES(COMPILE_SUM_DEVICE)
#undef COMPILE_SUM_DEVICE

}  // namespace math
}  // namespace magmadnn

#undef BLK_DIM
#undef BLK2D_DIM