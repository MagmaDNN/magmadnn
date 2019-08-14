/**
 * @file sgd_momentum_device.cu
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-26
 *
 * @copyright Copyright (c) 2019
 */
 #include "math/optimizer_math/sgd_momentum.h"

 #define BLK_SIZE 1024
 
 namespace magmadnn {
 namespace math {
 
 template <typename T>
 __global__ void kernel_sgd_momentum_device(T learning_rate, T momentum, T *prev, T* grad, T *out, unsigned int size) {
     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int stride = blockDim.x * gridDim.x;
 
     for (unsigned int i = idx; i < size; i += stride) {
         prev[i] = momentum * prev[i] + (1 - momentum) * grad[i];
         out[i] = out[i] - learning_rate * prev[i];
     }
 }
 
 template <typename T>
 void sgd_momentum_device(T learning_rate, T momentum, Tensor<T> *prev, Tensor<T> *grad, Tensor<T> *out) {
     unsigned int size = out->get_size();
     kernel_sgd_momentum_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(learning_rate, momentum, prev->get_ptr(), 
                                                                                grad->get_ptr(), out->get_ptr(), size);
 }
 template void sgd_momentum_device(int learning_rate, int momentum, Tensor<int> *prev, Tensor<int> *grad, Tensor<int> *out);
 template void sgd_momentum_device(float learning_rate, float momentum, Tensor<float> *prev, Tensor<float> *grad, Tensor<float> *out);
 template void sgd_momentum_device(double learning_rate, double momentum, Tensor<double> *prev, Tensor<double> *grad, Tensor<double> *out);
 
 }  // namespace math
 }  // namespace magmadnn
 
 #undef BLK_SIZE