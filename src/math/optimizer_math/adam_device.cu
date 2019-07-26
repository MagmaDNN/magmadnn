/**
 * @file adam_device.cu
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-26
 *
 * @copyright Copyright (c) 2019
 */
 #include "math/optimizer_math/adam.h"

 #define BLK_SIZE 1024
 
 namespace magmadnn {
 namespace math {
 
 template <typename T>
 __global__ void kernel_adam_device(T learning_rate, T beta1, T beta2, T running_beta1, T running_beta2, 
     T *first_moment, T *second_moment, T *grad, T *out) {
     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int stride = blockDim.x * gridDim.x;
 
     for (unsigned int i = idx; i < size; i += stride) {
         first_moment[i] = (beta1 * first_moment[i]) + (1 - beta1) * (grad[i]);
         second_moment[i] = (beta2 * second_moment[i]) + (1 - beta2) * (grad[i] * grad[i]);
         T m_temp = first_moment[i] / (1 - running_beta1);
         T v_temp = second_moment[i] / (1 - running_beta2);
         out[i] = out[i] - (learning_rate / (sqrt(v_temp) + 1e-8)) * m_temp;
     }
 }
 
 template <typename T>
 void adam_device(T learning_rate, T beta1, T beta2, T running_beta1, T running_beta2, 
    Tensor<T> *first_moment, Tensor<T> *second_moment, Tensor<T> *grad, Tensor<T> *out) {
     unsigned int size = out->get_size();
     kernel_adam_device<<<(size + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(learning_rate, beta1, beta2, running_beta1,
                                                                        running_beta2, first_moment->get_ptr, second_moment->get_ptr(), 
                                                                        grad->get_ptr(), out->get_ptr(), size);
 }
 template void adam_device(int learning_rate, int beta1, int beta2, int running_beta1, int running_beta2, 
    Tensor<int> *first_moment, Tensor<int> *second_moment, Tensor<int> *grad, Tensor<int> *out);
 template void adam_device(float learning_rate, float beta1, float beta2, float running_beta1, float running_beta2, 
    Tensor<float> *first_moment, Tensor<float> *second_moment, Tensor<float> *grad, Tensor<float> *out); 
 template void adam_device(double learning_rate, double beta1, double beta2, double running_beta1, double running_beta2, 
    Tensor<double> *first_moment, Tensor<double> *second_moment, Tensor<double> *grad, Tensor<double> *out); 
    
 }  // namespace math
 }  // namespace magmadnn
 
 #undef BLK_SIZE