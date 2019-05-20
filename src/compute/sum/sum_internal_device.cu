/**
 * @file sum_internal_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 * 
 * @copyright Copyright (c) 2019
 */
 #include "compute/sum/sum_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
__global__ void kernel_sum_full_device(T **arrs, unsigned int n_arrs, unsigned int arr_size, T *out) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    T sum;

    for (int i = idx; i < arr_size; i += stride) {
        sum = (T) 0;

        for (unsigned int j = 0; j < n_arrs; j++) {
            sum += arrs[j][i];
        }
        out[i] = sum;
    }
}


template <typename T>
void sum_full_device(std::vector<Tensor<T> *> &vals, Tensor<T> &out) {
    /* host array of device pointers */
    T **arrs_host;
    /* device array of device pointers */
    T **arrs_device;
    unsigned int n_arrs, arr_size;

    n_arrs = vals.size();
    arr_size = vals[0]->get_size();

    /* init arrs_host to hold array of device pointers for tensors */
    arrs_host = new T*[vals.size()];
    for (unsigned int i = 0; i < vals.size(); i++) {
        arrs_host[i] = vals[i]->get_ptr();
    }

    /* init arrs_device and copy device pointers into it */
    cudaMalloc((void **) &arrs_device, vals.size() * sizeof(T *));
    cudaMemcpy(arrs_device, arrs_host, vals.size() * sizeof(float *), cudaMemcpyHostToDevice);

    kernel_sum_full_device <<< n_arrs, arr_size >>> (arrs_device, n_arrs, arr_size, out.get_ptr());
    
    delete arrs_host;
    cudaFree(arrs_device);
}
template void sum_full_device(std::vector<Tensor<int> *> &vals, Tensor<int> &out);
template void sum_full_device(std::vector<Tensor<float> *> &vals, Tensor<float> &out);
template void sum_full_device(std::vector<Tensor<double> *> &vals, Tensor<double> &out);

}
}