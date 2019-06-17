/**
 * @file reduce_sum.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-14
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/reduce_sum.h"

namespace magmadnn {
namespace math {

namespace internal {

template <typename T>
void _gemv(T alpha, bool trans_A, Tensor<T> *A, Tensor<T> *x, T beta, Tensor<T> *out) {
    fprintf(stderr, "unimplemented _gemv type\n");
}

template <> void _gemv(float alpha, bool trans_A, Tensor<float> *A, Tensor<float> *x, float beta, Tensor<float> *out) {
    /* computes --   out = alpha*op( A ).x + beta*out */

    cblas_sgemv(CblasRowMajor,  /* data storage format */
        (trans_A) ? CblasTrans : CblasNoTrans,  /* op( A ) */
        A->get_shape(0),        /* M */
        A->get_shape(1),        /* N */
        alpha,                  /* alpha */
        A->get_ptr(),           /* ptr to A */
        A->get_shape(1),        /* leading dimension of A */
        x->get_ptr(),           /* ptr to x */
        1,                      /* stride of x */
        beta,                   /* beta */
        out->get_ptr(),         /* out */
        1
    );
}

template <> void _gemv(double alpha, bool trans_A, Tensor<double> *A, Tensor<double> *x, double beta, Tensor<double> *out) {
    /* computes --   out = alpha*op( A ).x + beta*out */

    cblas_dgemv(CblasRowMajor,  /* data storage format */
        (trans_A) ? CblasTrans : CblasNoTrans,  /* op( A ) */
        A->get_shape(0),        /* M */
        A->get_shape(1),        /* N */
        alpha,                  /* alpha */
        A->get_ptr(),           /* ptr to A */
        A->get_shape(1),        /* leading dimension of A */
        x->get_ptr(),           /* ptr to x */
        1,                      /* stride of x */
        beta,                   /* beta */
        out->get_ptr(),         /* out */
        1
    );
}

}   // internal

template <typename T>
void reduce_sum(Tensor<T> *x, int axis, Tensor<T> *ones, Tensor<T> *out) {
    if (out->get_memory_type() == HOST) {
        /* compute sum on CPU */
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();

        if (T_IS_VECTOR(x) || T_IS_SCALAR(x)) {
            /* simple sum all the elements of x */
            unsigned int size = x->get_size();
            out_ptr[0] = (T) 0;
            for (unsigned int i = 0; i < size; i++) {
                out_ptr[0] += x_ptr[i];
            }            
        } else if (T_IS_MATRIX(x)) {
            /* use gemv to compute row-sum or col-sum */
            if (axis == 0) {
                /* call to external templated functions */
                internal::_gemv((T)1, true, x, ones, (T)0, out); // column reduce
            } else {
                internal::_gemv((T)1, false, x, ones, (T)0, out); // row reduce
            }
        } else {
            /* sum tensor axis */
            fprintf(stderr, ">= 3D tensor reduction not yet support on CPU.\n");
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "Please use reduce_sum_device for GPU sum_reduce.\n");
    }
    #endif
}
template void reduce_sum(Tensor<int> *x, int axis, Tensor<int> *ones, Tensor<int> *out);
template void reduce_sum(Tensor<float> *x, int axis, Tensor<float> *ones, Tensor<float> *out);
template void reduce_sum(Tensor<double> *x, int axis, Tensor<double> *ones, Tensor<double> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void reduce_sum_device(Tensor<T> *x, int axis, Tensor<T> *out, reduce_sum_cudnn_settings_t settings) {
    /* call cudnn */
    T alpha = (T) 1;
    T beta = (T) 0;
    cudnnErrchk( cudnnReduceTensor(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle,
        settings.descriptor,
        NULL,       /* indices ptr */
        0,          /* indices ptr size */
        settings.workspace,
        settings.workspace_size,
        &alpha,
        x->get_cudnn_tensor_descriptor(),
        x->get_ptr(),
        &beta, 
        out->get_cudnn_tensor_descriptor(),
        out->get_ptr()
    ) );
}
template void reduce_sum_device(Tensor<int> *x, int axis, Tensor<int> *out, reduce_sum_cudnn_settings_t settings);
template void reduce_sum_device(Tensor<float> *x, int axis, Tensor<float> *out, reduce_sum_cudnn_settings_t settings);
template void reduce_sum_device(Tensor<double> *x, int axis, Tensor<double> *out, reduce_sum_cudnn_settings_t settings);
#endif

}
}