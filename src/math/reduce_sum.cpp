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
void _gemv(T alpha, bool trans_A, const Tensor &A, const Tensor &x, T beta, Tensor &out) {
    LOG(ERROR) << "unimplemented _gemv type\n";
}

template <>
void _gemv(float alpha, bool trans_A, const Tensor &A, const Tensor &x, float beta, Tensor &out) {
    /* computes --   out = alpha*op( A ).x + beta*out */

    cblas_sgemv(CblasRowMajor,                         /* data storage format */
                (trans_A) ? CblasTrans : CblasNoTrans, /* op( A ) */
                A.shape(0),                            /* M */
                A.shape(1),                            /* N */
                alpha,                                 /* alpha */
                A.get_ptr<float>(),                    /* ptr to A */
                A.shape(1),                            /* leading dimension of A */
                x.get_ptr<float>(),                    /* ptr to x */
                1,                                     /* stride of x */
                beta,                                  /* beta */
                out.get_ptr<float>(),                  /* out */
                1);
}

template <>
void _gemv(double alpha, bool trans_A, const Tensor &A, const Tensor &x, double beta, Tensor &out) {
    /* computes --   out = alpha*op( A ).x + beta*out */

    cblas_dgemv(CblasRowMajor,                         /* data storage format */
                (trans_A) ? CblasTrans : CblasNoTrans, /* op( A ) */
                A.shape(0),                            /* M */
                A.shape(1),                            /* N */
                alpha,                                 /* alpha */
                A.get_ptr<double>(),                   /* ptr to A */
                A.shape(1),                            /* leading dimension of A */
                x.get_ptr<double>(),                   /* ptr to x */
                1,                                     /* stride of x */
                beta,                                  /* beta */
                out.get_ptr<double>(),                 /* out */
                1);
}

}  // namespace internal

template <typename T>
void reduce_sum(const Tensor &x, int axis, const Tensor &ones, Tensor &out) {
    MAGMADNN_ASSERT(
        ::magmadnn::utilities::do_tensors_match(GetDataType<T>::value, out.get_memory_type(), {x, ones, out}),
        "invalid tensors");

    if (out.get_memory_type() == HOST) {
        /* compute sum on CPU */
        const T *x_ptr = x.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();

        /* scalar or vector */
        if ((x.size() == 1) || (x.shape().size() == 1)) {
            /* simple sum all the elements of x */
            size_t size = x.size();
            out_ptr[0] = static_cast<T>(0);
            for (unsigned int i = 0; i < size; i++) {
                out_ptr[0] += x_ptr[i];
            }
        } else if (x.shape().size() == 2) {
            /* use gemv to compute row-sum or col-sum */
            if (axis == 0) {
                /* call to external templated functions */
                /* axis == 0 -- col sum */
                internal::_gemv(static_cast<T>(1), true, x, ones, static_cast<T>(0), out);  // column reduce
            } else {
                /* axis == 1 -- row sum */
                internal::_gemv(static_cast<T>(1), false, x, ones, static_cast<T>(0), out);  // row reduce
            }
        } else {
            /* sum tensor axis */
            LOG(ERROR) << ">= 3D tensor reduction not yet support on CPU.\n";
        }
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "Please use reduce_sum_device for GPU sum_reduce.\n";
    }
#endif
}
#define comp(type) template void reduce_sum<type>(const Tensor &, int, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(comp)
#undef comp

#if defined(_HAS_CUDA_)
template <typename T>
void reduce_sum_device(const Tensor &x, int axis, Tensor &out, reduce_sum_cudnn_settings_t settings) {
    /* call cudnn */
    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);

    /* if scalar, just copy */
    if (x.size() == out.size()) {
        out.copy_from(x);
    }

    /* else do reduce sum */
    else {
        cudnnErrchk(cudnnReduceTensor(::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, /* cudnn handle */
                                      settings.descriptor,               /* reduce tensor descriptor */
                                      NULL,                              /* indices ptr */
                                      0,                                 /* indices ptr size */
                                      settings.workspace,                /* ptr to workspace */
                                      settings.workspace_size,           /* size of memory allocated to workspace ptr */
                                      &alpha,                            /* alpha */
                                      x.get_cudnn_tensor_descriptor(),   /* x -- descriptor */
                                      x.get_ptr<T>(),                    /* x ptr */
                                      &beta,                             /* beta */
                                      out.get_cudnn_tensor_descriptor(), /* out -- descriptor */
                                      out.get_ptr<T>()                   /*out ptr */
                                      ));
    }
}
#define comp(type) template void reduce_sum_device<type>(const Tensor &, int, Tensor &, reduce_sum_cudnn_settings_t);
CALL_FOR_ALL_TYPES(comp)
#undef comp

#endif

}  // namespace math
}  // namespace magmadnn