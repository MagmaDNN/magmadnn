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
__global__ void kernel_sum_device(T** tensors, T* out, unsigned int size, unsigned int n_tensors) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        out[i] = tensors[0][i];
    }

    /* x_idx will span the individual tensor sizes */
    for (unsigned int t = 1; t < n_tensors; t++) {
        for (unsigned int i = idx; i < size; i += stride) {
            out[i] += tensors[t][i];
        }
    }
}

template <typename T>
void sum_device(const std::vector<Tensor<T>*>& tensors, Tensor<T>* out) {
    unsigned int size = out->get_size();

    T** tensors_arr, **host_tensors_arr;
    host_tensors_arr = (T**) malloc(tensors.size() * sizeof(T*));
    cudaErrchk(cudaMalloc((void**) &tensors_arr, tensors.size() * sizeof(T*)));
    
    for (unsigned int i = 0; i < tensors.size(); i++) {
        host_tensors_arr[i] = tensors[i]->get_ptr();
    }
    
    cudaErrchk(cudaMemcpy(tensors_arr, host_tensors_arr, tensors.size() * sizeof(T*), cudaMemcpyHostToDevice));

    kernel_sum_device<<<(size+BLK_DIM-1)/BLK_DIM,BLK_DIM>>> (tensors_arr, out->get_ptr(), size, tensors.size());

    cudaErrchk(cudaFree(tensors_arr));
    free(host_tensors_arr);
}
template void sum_device(const std::vector<Tensor<int>*>& tensors, Tensor<int>* out);
template void sum_device(const std::vector<Tensor<float>*>& tensors, Tensor<float>* out);
template void sum_device(const std::vector<Tensor<double>*>& tensors, Tensor<double>* out);

}  // namespace math
}  // namespace magmadnn

#undef BLK_DIM
#undef BLK2D_DIM