/**
 * @file pooling.cpp
 * @author Sedrick Keh
 * @author Rocco Febbo
 * @version 1.0
 * @date 2020-01-22
 *
 * @copyright Copyright (c) 2019
 */
#include "math/pooling.h"

#include <cassert>

#include "magmadnn/config.h"

namespace magmadnn {
namespace math {

template <typename T>
void pooling(Tensor<T> *x, Tensor<T> *out, Tensor<int> *max_positions, int filter_h, int filter_w, int pad_h, int pad_w,
             int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w, pooling_mode mode) {
    assert(T_IS_SAME_MEMORY_TYPE(x, out));

    if (out->get_memory_type() == HOST) {
        std::vector<unsigned int> out_shape = out->get_shape();
        std::vector<unsigned int> in_shape = x->get_shape();

        unsigned int o_n, o_c, o_h, o_w, k_n, k_c, k_h, k_w;  // iteration variables
        unsigned int No, Co, Ho, Wo, Ck, Hk, Wk, Hi, Wi;      // shorthand for tensor dims
        unsigned int total_filter_vals = filter_w * filter_h;
        if (out_shape.size() == 4) {
            No = out_shape[0];
            Co = out_shape[1];
            Ho = out_shape[2];
            Wo = out_shape[3];
        } else if (out_shape.size() == 3) {
            No = 1;
            Co = out_shape[0];
            Ho = out_shape[1];
            Wo = out_shape[2];
        } else if (out_shape.size() == 2) {
            No = 1;
            Co = 1;
            Ho = out_shape[0];
            Wo = out_shape[1];
        } else {
            fprintf(stderr, "Error: pooling::invalid output shape size.\n");
            return;
        }

        if (in_shape.size() == 4) {
            Hi = in_shape[2];
            Wi = in_shape[3];
        } else if (in_shape.size() == 3) {
            Hi = in_shape[1];
            Wi = in_shape[2];
        } else if (in_shape.size() == 2) {
            Hi = in_shape[0];
            Wi = in_shape[1];
        } else {
            fprintf(stderr, "Error: pooling::invalid input shape size.\n");
            return;
        }

        for (o_n = 0; o_n < No; o_n++) {
            for (o_c = 0; o_c < Co; o_c++) {
                for (o_h = 0; o_h < Ho; o_h++) {
                    for (o_w = 0; o_w < Wo; o_w++) {
                        float val = 0;
                        bool set = false;

                        // traverse filter
                        for (k_h = 0; k_h < filter_h; k_h++) {
                            for (k_w = 0; k_w < filter_w; k_w++) {
                                unsigned int in_h = o_h * vertical_stride - pad_h + k_h * dilation_h;
                                unsigned int in_w = o_w * horizontal_stride - pad_w + k_w * dilation_w;

                                if (in_h < Hi && in_w < Wi) {
                                    if (mode == pooling_mode::MAX_POOL) {
                                        float tmp = x->get({o_n, k_c, in_h, in_w});
                                        if (tmp > val || !set) val = tmp;
                                    } else if (mode == pooling_mode::AVERAGE_POOL) {
                                        val += x->get({o_n, k_c, in_h, in_w}) / total_filter_vals;
                                    }
                                }
                            }
                        }

                        if (mode == pooling_mode::MAX_POOL && val != 0) max_positions->set({o_n, o_c, o_h, o_w}, 1);

                        out->set({o_n, o_c, o_h, o_w}, val);
                    }
                }
            }
        }
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For Pooling GPU please use pooling_device.\n");
    }
#endif
}
template void pooling(Tensor<int> *x, Tensor<int> *out, Tensor<int> *max_positions, int filter_h, int filter_w,
                      int pad_h, int pad_w, int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                      pooling_mode mode);
template void pooling(Tensor<float> *x, Tensor<float> *out, Tensor<int> *max_positions, int filter_h, int filter_w,
                      int pad_h, int pad_w, int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                      pooling_mode mode);
template void pooling(Tensor<double> *x, Tensor<double> *out, Tensor<int> *max_positions, int filter_h, int filter_w,
                      int pad_h, int pad_w, int vertical_stride, int horizontal_stride, int dilation_h, int dilation_w,
                      pooling_mode mode);

template <typename T>
void pooling_grad(Tensor<T> *x, Tensor<T> *y, Tensor<T> *grad, Tensor<int> *max_positions, Tensor<T> *out, int filter_h,
                  int filter_w, int pad_h, int pad_w, int vertical_stride, int horizontal_stride, int dilation_h,
                  int dilation_w, pooling_mode mode) {
    assert(T_IS_SAME_MEMORY_TYPE(x, y));
    assert(T_IS_SAME_MEMORY_TYPE(y, grad));
    assert(T_IS_SAME_MEMORY_TYPE(grad, out));

    if (out->get_memory_type() == HOST) {
        fprintf(stderr, "__Pooling_grad CPU not supported yet.\n");
        std::vector<unsigned int> y_shape = y->get_shape();
        std::vector<unsigned int> out_shape = out->get_shape();

        unsigned int y_n, y_c, y_h, y_w, k_n, k_c, k_h, k_w;          // iteration variables
        unsigned int No, Co, Ho, Wo, Ny, Cy, Hy, Wy, Ck, Hk, Wk, Ci;  // shorthand for tensor dims
        unsigned int total_filter_vals = filter_w * filter_h;
        if (y_shape.size() == 4) {
            Ny = y_shape[0];
            Cy = y_shape[1];
            Hy = y_shape[2];
            Wy = y_shape[3];
        } else if (y_shape.size() == 3) {
            Ny = 1;
            Cy = y_shape[0];
            Hy = y_shape[1];
            Wy = y_shape[2];
        } else if (y_shape.size() == 2) {
            Ny = 1;
            Cy = 1;
            Hy = y_shape[0];
            Wy = y_shape[1];
        } else {
            fprintf(stderr, "Error: pooling::invalid output shape size.\n");
            return;
        }

        if (out_shape.size() == 4) {
            No = out_shape[0];
            Co = out_shape[1];
            Ho = out_shape[2];
            Wo = out_shape[3];
        } else if (out_shape.size() == 3) {
            No = 1;
            Co = out_shape[0];
            Ho = out_shape[1];
            Wo = out_shape[2];
        } else if (out_shape.size() == 2) {
            No = 1;
            Co = 1;
            Ho = out_shape[1];
            Wo = out_shape[2];
        } else {
            fprintf(stderr, "Error: pooling::invalid output shape size.\n");
            return;
        }

        for (y_n = 0; y_n < Ny; y_n++) {
            for (y_c = 0; y_c < Cy; y_c++) {
                for (y_h = 0; y_h < Hy; y_h++) {
                    for (y_w = 0; y_w < Wy; y_w++) {
                        float val = 0;
                        bool set = false;

                        // traverse filter
                        for (k_h = 0; k_h < filter_h; k_h++) {
                            for (k_w = 0; k_w < filter_w; k_w++) {
                                unsigned int out_h = y_h * vertical_stride - pad_h + k_h * dilation_h;
                                unsigned int out_w = y_w * horizontal_stride - pad_w + k_w * dilation_w;

                                if (out_h < Ho && out_w < Wo) {
                                    if (mode == pooling_mode::MAX_POOL) {
                                        val =
                                            y->get({y_n, y_c, y_h, y_w}) * max_positions->get({y_n, k_c, out_h, out_w});
                                        out->set({y_n, k_c, out_h, out_w}, val);

                                    } else if (mode == pooling_mode::AVERAGE_POOL) {
                                        val = y->get({y_n, y_c, y_h, y_w}) / total_filter_vals;
                                        out->set({y_n, k_c, out_h, out_w}, val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#if defined(MAGMADNN_HAVE_CUDA)
    else {
        fprintf(stderr, "For pooling_grad GPU please use pooling_grad_device.\n");
    }
#endif
}
template void pooling_grad(Tensor<int> *x, Tensor<int> *y, Tensor<int> *grad, Tensor<int> *max_positions,
                           Tensor<int> *out, int filter_h, int filter_w, int pad_h, int pad_w, int vertical_stride,
                           int horizontal_stride, int dilation_h, int dilation_w, pooling_mode mode);
template void pooling_grad(Tensor<float> *x, Tensor<float> *y, Tensor<float> *grad, Tensor<int> *max_positions,
                           Tensor<float> *out, int filter_h, int filter_w, int pad_h, int pad_w, int vertical_stride,
                           int horizontal_stride, int dilation_h, int dilation_w, pooling_mode mode);
template void pooling_grad(Tensor<double> *x, Tensor<double> *y, Tensor<double> *grad, Tensor<int> *max_positions,
                           Tensor<double> *out, int filter_h, int filter_w, int pad_h, int pad_w, int vertical_stride,
                           int horizontal_stride, int dilation_h, int dilation_w, pooling_mode mode);

#if defined(MAGMADNN_HAVE_CUDA)

template <typename T>
void pooling_device(Tensor<T> *x, Tensor<T> *out, cudnn_pooling_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);

    cudnnErrchk(cudnnPoolingForward(settings.handle, settings.poolingDesc, &alpha, x->get_cudnn_tensor_descriptor(),
                                    x->get_ptr(), &beta, out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}
template void pooling_device(Tensor<int> *x, Tensor<int> *out, cudnn_pooling_settings_t settings);
template void pooling_device(Tensor<float> *x, Tensor<float> *out, cudnn_pooling_settings_t settings);
template void pooling_device(Tensor<double> *x, Tensor<double> *out, cudnn_pooling_settings_t settings);

template <typename T>
void pooling_grad_device(Tensor<T> *x, Tensor<T> *y, Tensor<T> *grad, Tensor<T> *out,
                         cudnn_pooling_settings_t settings) {
    T alpha = static_cast<T>(1), beta = static_cast<T>(0);

    cudnnErrchk(cudnnPoolingBackward(settings.handle, settings.poolingDesc, &alpha, y->get_cudnn_tensor_descriptor(),
                                     y->get_ptr(), grad->get_cudnn_tensor_descriptor(), grad->get_ptr(),
                                     x->get_cudnn_tensor_descriptor(), x->get_ptr(), &beta,
                                     out->get_cudnn_tensor_descriptor(), out->get_ptr()));
}

template void pooling_grad_device(Tensor<int> *x, Tensor<int> *y, Tensor<int> *grad, Tensor<int> *out,
                                  cudnn_pooling_settings_t settings);
template void pooling_grad_device(Tensor<float> *x, Tensor<float> *y, Tensor<float> *grad, Tensor<float> *out,
                                  cudnn_pooling_settings_t settings);
template void pooling_grad_device(Tensor<double> *x, Tensor<double> *y, Tensor<double> *grad, Tensor<double> *out,
                                  cudnn_pooling_settings_t settings);

#endif

}  // namespace math
}  // namespace magmadnn
