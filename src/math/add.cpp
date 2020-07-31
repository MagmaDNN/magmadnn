/**
 * @file add.cpp
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-06-24
 *
 * @copyright Copyright (c) 2019
 */
#include "math/add.h"

#include <cassert>

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "math/wrappers.h"

namespace magmadnn {
namespace math {

////////////////////////////////////////
// out = alpha * x + beta * out
// CPU
template <typename T>
void add_in_place_cpu(T alpha, Tensor<T> *x, T beta, Tensor<T> *out) {
    T *x_ptr = x->get_ptr();
    T *out_ptr = out->get_ptr();
    unsigned int size = out->get_size();

    for (unsigned int i = 0; i < size; i++) {
        out_ptr[i] = alpha * x_ptr[i] + beta * out_ptr[i];
    }
}

template void add_in_place_cpu(int alpha, Tensor<int> *x, int beta, Tensor<int> *out);
template void add_in_place_cpu(float alpha, Tensor<float> *x, float beta, Tensor<float> *out);
template void add_in_place_cpu(double alpha, Tensor<double> *x, double beta, Tensor<double> *out);

////////////////////////////////////////
// out = alpha * x + out
// CPU
template <typename T>
void add_in_place_cpu(T alpha, Tensor<T> *x, Tensor<T> *out) {
    T const *x_ptr = x->get_ptr();
    T *out_ptr = out->get_ptr();
    unsigned int size = out->get_size();

    axpy(size, alpha, x_ptr, 1, out_ptr, 1);
}

template void add_in_place_cpu(float alpha, Tensor<float> *x, Tensor<float> *out);
template void add_in_place_cpu(double alpha, Tensor<double> *x, Tensor<double> *out);

template <>
void add_in_place_cpu<int>(int alpha, Tensor<int> *x, Tensor<int> *out) {
    add_in_place_cpu(alpha, x, static_cast<int>(1), out);
}

////////////////////////////////////////
// out = x + out
// CPU
template <typename T>
void add_in_place_cpu(Tensor<T> *x, Tensor<T> *out) {
    add_in_place_cpu(static_cast<T>(1.0), x, out);
}

template void add_in_place_cpu(Tensor<int> *x, Tensor<int> *out);
template void add_in_place_cpu(Tensor<float> *x, Tensor<float> *out);
template void add_in_place_cpu(Tensor<double> *x, Tensor<double> *out);

////////////////////////////////////////
// out = out - x
// CPU
template <typename T>
void subtract_cpu(Tensor<T> *x, Tensor<T> *out) {
    add_in_place_cpu(static_cast<T>(-1.0), x, out);
}

template void subtract_cpu(Tensor<int> *x, Tensor<int> *out);
template void subtract_cpu(Tensor<float> *x, Tensor<float> *out);
template void subtract_cpu(Tensor<double> *x, Tensor<double> *out);

////////////////////////////////////////
// out = alpha * x + beta * out
// CPU & GPU
template <typename T>
void add_in_place(T alpha, Tensor<T> *x, T beta, Tensor<T> *out) {
    assert(x->get_size() == out->get_size());
    assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out->get_memory_type() == HOST) {
        add_in_place_cpu(alpha, x, beta, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        add_in_place_device(alpha, x, beta, out);
    }
#endif
}
template void add_in_place(int alpha, Tensor<int> *x, int beta, Tensor<int> *out);
template void add_in_place(float alpha, Tensor<float> *x, float beta, Tensor<float> *out);
template void add_in_place(double alpha, Tensor<double> *x, double beta, Tensor<double> *out);

template <typename T>
void add_in_place(T alpha, Tensor<T> *x, Tensor<T> *out) {
    assert(x->get_size() == out->get_size());
    assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out->get_memory_type() == HOST) {
        add_in_place_cpu(alpha, x, out);
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        add_in_place_device(alpha, x, static_cast<T>(1.0), out);
    }
#endif
}
template void add_in_place(int alpha, Tensor<int> *x, Tensor<int> *out);
template void add_in_place(float alpha, Tensor<float> *x, Tensor<float> *out);
template void add_in_place(double alpha, Tensor<double> *x, Tensor<double> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void add_in_place_device(cudnnHandle_t handle, T alpha, Tensor<T> *x, T beta, Tensor<T> *out) {
    cudnnErrchk(cudnnAddTensor(handle, &alpha, x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
                               out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}

template void add_in_place_device(cudnnHandle_t handle, int alpha, Tensor<int> *x, int beta, Tensor<int> *out);
template void add_in_place_device(cudnnHandle_t handle, float alpha, Tensor<float> *x, float beta, Tensor<float> *out);
template void add_in_place_device(cudnnHandle_t handle, double alpha, Tensor<double> *x, double beta,
                                  Tensor<double> *out);

template <typename T>
void add_in_place_device(cudnnHandle_t handle, Tensor<T> *x, Tensor<T> *out) {
    add_in_place_device(handle, static_cast<T>(1.0), x, static_cast<T>(1.0), out);
}

template void add_in_place_device(cudnnHandle_t handle, Tensor<int> *x, Tensor<int> *out);
template void add_in_place_device(cudnnHandle_t handle, Tensor<float> *x, Tensor<float> *out);
template void add_in_place_device(cudnnHandle_t handle, Tensor<double> *x, Tensor<double> *out);

////////////////////////////////////////
// out = out - x
// CUDA GPU
template <typename T>
void subtract_device(cudnnHandle_t handle, Tensor<T> *x, Tensor<T> *out) {
    add_in_place_device(handle, static_cast<T>(-1.0), x, static_cast<T>(1.0), out);
}

template void subtract_device(cudnnHandle_t handle, Tensor<int> *x, Tensor<int> *out);
template void subtract_device(cudnnHandle_t handle, Tensor<float> *x, Tensor<float> *out);
template void subtract_device(cudnnHandle_t handle, Tensor<double> *x, Tensor<double> *out);

template <typename T>
void add_in_place_device(T alpha, Tensor<T> *x, T beta, Tensor<T> *out) {
    cudnnErrchk(cudnnAddTensor(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, &alpha,
                               x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
                               out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}

template void add_in_place_device(int alpha, Tensor<int> *x, int beta, Tensor<int> *out);
template void add_in_place_device(float alpha, Tensor<float> *x, float beta, Tensor<float> *out);
template void add_in_place_device(double alpha, Tensor<double> *x, double beta, Tensor<double> *out);
#endif

}  // namespace math
}  // namespace magmadnn
