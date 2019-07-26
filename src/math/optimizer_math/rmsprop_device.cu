/**
 * @file rmsprop_device.cu
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-26
 *
 * @copyright Copyright (c) 2019
 */
 #include "math/optimizer_math/rmsprop.h"

 #define BLK_SIZE 1024
 
 namespace magmadnn {
 namespace math {
 
 template <typename T>
 __global__ void kernel_rmsprop_device(T learning_rate, T decaying_factor, T *decaying_squares_average, 
    T* grad, T *out, unsigned int size) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        decaying_squares_average[i] = (decaying_factor * decaying_squares_average[i]) +
                                      (1 - decaying_factor) * (grad[i] * grad[i]);
        out[i] = out[i] - (learning_rate / sqrt(1e-8 + decaying_squares_average[i])) * grad[i];
    }
}
 
 template <typename T>
 void rmsprop_device(T learning_rate, T decaying_factor, Tensor<T> *decaying_squares_average, Tensor<T> *grad, Tensor<T> *out) {
     unsigned int size = out->get_size();
     kernel_rmsprop_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(learning_rate, decaying_factor, decaying_squares_average->get_ptr(), 
                                                                                grad->get_ptr(), out->get_ptr(), size);
 }
 template void rmsprop_device(int learning_rate, int decaying_factor, Tensor<int> *decaying_squares_average, Tensor<int> *grad, Tensor<int> *out);
 template void rmsprop_device(float learning_rate, float decaying_factor, Tensor<float> *decaying_squares_average, Tensor<float> *grad, Tensor<float> *out);
 template void rmsprop_device(double learning_rate, double decaying_factor, Tensor<double> *decaying_squares_average, Tensor<double> *grad, Tensor<double> *out);
 
 }  // namespace math
 }  // namespace magmadnn
 
 #undef BLK_SIZE