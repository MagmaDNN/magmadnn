/**
 * @file adagrad.cu
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-26
 *
 * @copyright Copyright (c) 2019
 */
 #include "math/optimizer_math/adagrad.h"

 #define BLK_SIZE 1024
 
 namespace magmadnn {
 namespace math {
 
 template <typename T>
 __global__ void kernel_adagrad_device(T learning_rate, T *scaling_tensors, T* grad, T *out, unsigned int size) {
     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int stride = blockDim.x * gridDim.x;
 
     for (unsigned int i = idx; i < size; i += stride) {
         scaling_tensors[i] += (grad[i] * grad[i]);
         out[i] = out[i] - (learning_rate / sqrt(1e-8 + scaling_tensors[i])) * grad[i];
     }
 }
 
 template <typename T>
 void adagrad_device(T learning_rate, Tensor<T> *scaling_tensors, Tensor<T> *grad, Tensor<T> *out) {
     unsigned int size = out->get_size();
     kernel_adagrad_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(learning_rate, scaling_tensors->get_ptr(), 
                                                                                grad->get_ptr(), out->get_ptr(), size);
 }
 template void adagrad_device(int learning_rate, Tensor<int> *scaling_tensors, Tensor<int> *grad, Tensor<int> *out);
 template void adagrad_device(float learning_rate, Tensor<float> *scaling_tensors, Tensor<float> *grad, Tensor<float> *out);
 template void adagrad_device(double learning_rate, Tensor<double> *scaling_tensors, Tensor<double> *grad, Tensor<double> *out);
 
 }  // namespace math
 }  // namespace magmadnn
 
 #undef BLK_SIZE