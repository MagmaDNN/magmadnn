
#include "compute/reducesum/reducesum_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void tensor_reducesum_full(Tensor<T> *x, unsigned int axis, Tensor<T> *out) {

    if (out->get_memory_type() == HOST) {
        
    }
    #if defined(_HAS_CUDA_)
    else {
        tensor_reducesum_full_device(x, axis, out);
    }
    #endif

}
template void tensor_reducesum_full(Tensor<int> *x, unsigned int axis, Tensor<int> *out);
template void tensor_reducesum_full(Tensor<float> *x, unsigned int axis, Tensor<float> *out);
template void tensor_reducesum_full(Tensor<double> *x, unsigned int axis, Tensor<double> *out);


template <> void col_reducesum_full(Tensor<int> *x, Tensor<int> *ones, Tensor<int> *out) {
    
    unsigned int n_rows = x->get_shape(0);
    unsigned int n_cols = x->get_shape(1);
    unsigned int size = n_rows * n_cols;

    for (unsigned int i = 0; i < n_cols; i++) out->set(i, 0);

    for (unsigned int i = 0; i < size; i++) {
        out->set(i % n_cols, out->get(i % n_cols) + x->get(i));
    }
}
template <> void col_reducesum_full(Tensor<float> *x, Tensor<float> *ones, Tensor<float> *out) {
    if (out->get_memory_type() == HOST) {
        cblas_sgemv(CblasRowMajor, 
                    CblasTrans, 
                    x->get_shape(0), 
                    x->get_shape(1), 
                    (float) 1, 
                    ones->get_ptr(), 
                    x->get_shape(1),
                    out->get_ptr(),
                    1,
                    (float) 0,
                    out->get_ptr(),
                    1);
    }
    #if defined(_HAS_CUDA_)
    else {
        /* gemv to col reduce */
        magma_sgemv(MagmaTrans,
                    x->get_shape(0),
                    x->get_shape(1),
                    (float) 1,
                    x->get_ptr(),
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float) 0,
                    out->get_ptr(),
                    1);
    }
    #endif
}
template <> void col_reducesum_full(Tensor<double> *x, Tensor<double> *ones, Tensor<double> *out) {
    if (out->get_memory_type() == HOST) {
        cblas_dgemv(CblasRowMajor, 
                    CblasTrans, 
                    x->get_shape(0), 
                    x->get_shape(1), 
                    (float) 1, 
                    x->get_ptr(), 
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float) 1,
                    out->get_ptr(),
                    1);
    }
    #if defined(_HAS_CUDA_)
    else {
        /* gemv to col reduce */
        magma_dgemv(MagmaTrans,
                    x->get_shape(0),
                    x->get_shape(1),
                    (float) 1,
                    x->get_ptr(),
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float) 0,
                    out->get_ptr(),
                    1);
    }
    #endif
}



template <> void row_reducesum_full(Tensor<int> *x, Tensor<int> *ones, Tensor<int> *out) {
    
    unsigned int n_rows = x->get_shape(0);
    unsigned int n_cols = x->get_shape(1);
    unsigned int size = n_rows * n_cols;

    for (unsigned int i = 0; i < n_rows; i++) out->set(i, 0);

    for (unsigned int i = 0; i < size; i++) {
        out->set(i / n_rows, out->get(i / n_rows) + x->get(i));
    }
}
template <> void row_reducesum_full(Tensor<float> *x, Tensor<float> *ones, Tensor<float> *out) {
    if (out->get_memory_type() == HOST) {
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    x->get_shape(0),
                    x->get_shape(1),
                    (float)1,
                    x->get_ptr(),
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float)0,
                    out->get_ptr(),
                    1);
    }
    #if defined(_HAS_CUDA_)
    else {
        magma_sgemv(MagmaNoTrans,
                    x->get_shape(1),
                    x->get_shape(0),
                    (float)1,
                    x->get_ptr(),
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float)0,
                    x->get_ptr(),
                    1);
    }
    #endif
}
template <> void row_reducesum_full(Tensor<double> *x, Tensor<double> *ones, Tensor<double> *out) {
    if (out->get_memory_type() == HOST) {
        cblas_dgemv(CblasRowMajor,
                    CblasNoTrans,
                    x->get_shape(0),
                    x->get_shape(1),
                    (float)1,
                    x->get_ptr(),
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float)0,
                    out->get_ptr(),
                    1);
    }
    #if defined(_HAS_CUDA_)
    else {
        magma_dgemv(MagmaNoTrans,
                    x->get_shape(1),
                    x->get_shape(0),
                    (float)1,
                    x->get_ptr(),
                    x->get_shape(1),
                    ones->get_ptr(),
                    1,
                    (float)0,
                    x->get_ptr(),
                    1);
    }
    #endif
}


template <typename T>
void reducesum_full(Tensor<T> *x, Tensor<T> *out) {

    if (out->get_memory_type() == HOST) {
        T *x_ptr = x->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int size = out->get_size();

        out_ptr[0] = (T) 0;
        for (unsigned int i = 0; i < size; i++) {
            out_ptr[0] += x_ptr[i];
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        reducesum_full_device(x, out);
    }
    #endif

}
template void reducesum_full(Tensor<int> *x, Tensor<int> *out);
template void reducesum_full(Tensor<float> *x, Tensor<float> *out);
template void reducesum_full(Tensor<double> *x, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn