#include "compute/gcnconv/gcnconvop.h"

namespace magmadnn {
namespace op {

#if defined(_HAS_CUDA_)
template <typename T>
void GCNConvOp<T>::cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, T alpha, T* A, int lda,
                                                long long strideA, T* B, int ldb, long long strideB, T beta, T* C,
                                                int ldc, long long strideC, int batch) {
    std::fprintf(stderr, "Unknown data type for GCNConvOp.\n");
}
template <>
void GCNConvOp<float>::cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, float alpha,
                                                    float* A, int lda, long long strideA, float* B, int ldb,
                                                    long long strideB, float beta, float* C, int ldc, long long strideC,
                                                    int batch) {
    cublasErrchk(cublasSgemmStridedBatched(::magmadnn::internal::MAGMADNN_SETTINGS->cublas_handle,
                                           trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, m,
                                           n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC,
                                           batch));
}
template <>
void GCNConvOp<double>::cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, double alpha,
                                                     double* A, int lda, long long strideA, double* B, int ldb,
                                                     long long strideB, double beta, double* C, int ldc,
                                                     long long strideC, int batch) {
    cublasErrchk(cublasDgemmStridedBatched(::magmadnn::internal::MAGMADNN_SETTINGS->cublas_handle,
                                           trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, m,
                                           n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC,
                                           batch));
}

template <typename T>
void GCNConvOp<T>::init_eval_cublas(void) {
    this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    cTbT_tensor = new Tensor<T>({n_samples, n_channel_out, n_vert_in}, this->mem_type);
}
template <typename T>
void GCNConvOp<T>::init_gTa_cublas(void) {
    gTa_tensor = new Tensor<T>({n_samples, n_channel_out, n_vert_in}, this->mem_type);
}
template <typename T>
void GCNConvOp<T>::init_bTaTg_cublas(void) {
    bTaTg_tensor = new Tensor<T>({n_samples, n_channel_in * n_channel_out}, this->mem_type);
    ones = new Tensor<T>({1, n_samples}, {ONE, {}}, this->mem_type);
}
#else
template <typename T>
void GCNConvOp<T>::init_eval(void) {
    this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    b_tensor_slice = new Tensor<T>(input_sample_size, this->mem_type);
    ab_tensor_slice = new Tensor<T>({n_vert_out, n_channel_in}, this->mem_type);
    abc_tensor_slice = new Tensor<T>({n_vert_out, n_channel_out}, this->mem_type);
}
template <typename T>
void GCNConvOp<T>::init_grad(void) {
    grad_tensor_slice = new Tensor<T>({n_vert_out, n_channel_out}, this->mem_type);
    aTg_tensor_slice = new Tensor<T>({n_vert_in, n_channel_out}, this->mem_type);
}

#endif

template <typename T>
GCNConvOp<T>::GCNConvOp(Tensor<T>* a, Operation<T>* b, Operation<T>* c, bool copy, bool needs_grad)
    : Operation<T>({b, c}, needs_grad),
      a(a),
      b(b),
      c(c),
      copy(copy),
      b_tensor(nullptr),
      b_tensor_slice(nullptr),
      c_tensor(nullptr),
      ab_tensor_slice(nullptr),
      abc_tensor_slice(nullptr),
      grad_tensor_slice(nullptr),
      aTg_tensor_slice(nullptr),

      cTbT_tensor(nullptr),
      gTa_tensor(nullptr),
      bTaTg_tensor(nullptr),
      ones(nullptr) {
    assert(a->get_memory_type() == b->get_memory_type());
    assert(OP_IS_SAME_MEMORY_TYPE(b, c));
    assert(T_IS_MATRIX(a));
    assert(OP_IS_N_DIMENSIONAL(b, 3));
    assert(OP_IS_MATRIX(c));
    assert(a->get_shape(1) == b->get_output_shape(1));
    assert(b->get_output_shape(2) == c->get_output_shape(0));
    n_samples = b->get_output_shape(0);
    n_vert_in = a->get_shape(1);
    n_vert_out = a->get_shape(0);
    n_channel_in = c->get_output_shape(0);
    n_channel_out = c->get_output_shape(1);
    input_sample_size = n_vert_in * n_channel_in;
    output_sample_size = n_vert_out * n_channel_out;
    this->output_shape = {n_samples, n_vert_out, n_channel_out};
    this->mem_type = a->get_memory_type();
}
template <typename T>
GCNConvOp<T>::~GCNConvOp(void) {
    delete_if_not_nullptr(b_tensor);
    delete_if_not_nullptr(b_tensor_slice);
    delete_if_not_nullptr(c_tensor);
    delete_if_not_nullptr(ab_tensor_slice);
    delete_if_not_nullptr(abc_tensor_slice);
    delete_if_not_nullptr(grad_tensor_slice);
    delete_if_not_nullptr(aTg_tensor_slice);
    delete_if_not_nullptr(cTbT_tensor);
    delete_if_not_nullptr(gTa_tensor);
    delete_if_not_nullptr(bTaTg_tensor);
    delete_if_not_nullptr(ones);
}
template <typename T>
Tensor<T>* GCNConvOp<T>::_eval(bool recompute) {
    b_tensor = b->eval(recompute);
    c_tensor = c->eval(recompute);
//  compute c_tensor^T * b_tensor_slice^T, stored in cTbT_tensor in column-major (so in row-major it is b_slice * c)
#if defined(_HAS_CUDA_)
    if (cTbT_tensor == nullptr) {
        init_eval_cublas();
    }
    this->cublasGemmStridedBatchedWrap(false, false, n_channel_out, n_vert_in, n_channel_in, const_one,
                                       c_tensor->get_ptr(), n_channel_out, 0, b_tensor->get_ptr(), n_channel_in,
                                       input_sample_size, const_zero, cTbT_tensor->get_ptr(), n_channel_out,
                                       n_vert_in * n_channel_out, n_samples);
    //  compute cTbT_tensor * a^T, stored in out in column-major (so in row-major it is a * b_slice * c)
    this->cublasGemmStridedBatchedWrap(false, false, n_channel_out, n_vert_out, n_vert_in, const_one,
                                       cTbT_tensor->get_ptr(), n_channel_out, n_channel_out * n_vert_in, a->get_ptr(),
                                       n_vert_in, 0, const_zero, this->output_tensor->get_ptr(), n_channel_out,
                                       output_sample_size, n_samples);
#else
    if (b_tensor_slice == nullptr) {
        init_eval();
    }
    for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {  //  for each sample
        //  compute a * b_tensor_slice * c, put into output
        b_tensor_slice->copy_from(*b_tensor, sample_idx * input_sample_size, input_sample_size);
        math::matmul(const_one, false, a, false, b_tensor_slice, const_zero, ab_tensor_slice);
        math::matmul(const_one, false, ab_tensor_slice, false, c_tensor, const_zero, abc_tensor_slice);
        this->output_tensor->copy_from(*this->abc_tensor_slice, 0, output_sample_size, sample_idx * output_sample_size);
    }
#endif
    return this->output_tensor;
}
template <typename T>
Tensor<T>* GCNConvOp<T>::_grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad) {
    assert(T_IS_N_DIMENSIONAL(grad, 3));
    assert(grad->get_shape(0) == n_samples);
    assert(grad->get_shape(1) == n_vert_out);
    assert(grad->get_shape(2) == n_channel_out);
    Tensor<T>* out = this->_grad_cache[(uintptr_t) var];
    if (var == b) {
        //  out_{i} = a^T * grad_{i} * c^T
        this->c_tensor = c->eval(false);
        if (out == NULL) {
            out = new Tensor<T>(b->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) b] = out;
        }
#if defined(_HAS_CUDA_)
        if (gTa_tensor == nullptr) {
            init_gTa_cublas();
        }
        //  compute grad^T * (a^T)^T , stored in gTa_tensor in column-major (so in row-major it is a^T * grad)
        this->cublasGemmStridedBatchedWrap(false, true, n_channel_out, n_vert_in, n_vert_out, const_one,
                                           grad->get_ptr(), n_channel_out, output_sample_size, a->get_ptr(), n_vert_in,
                                           0, const_zero, gTa_tensor->get_ptr(), n_channel_out,
                                           n_channel_out * n_vert_in, n_samples);
        //  compute (c^T)^T * gTa , stored in out in column-major (so in row-major it is a^T * grad * c^T)
        this->cublasGemmStridedBatchedWrap(true, false, n_channel_in, n_vert_in, n_channel_out, const_one,
                                           c_tensor->get_ptr(), n_channel_out, 0, gTa_tensor->get_ptr(), n_channel_out,
                                           n_channel_out * n_vert_in, const_zero, out->get_ptr(), n_channel_in,
                                           input_sample_size, n_samples);
#else
        if (grad_tensor_slice == nullptr) {
            init_grad();
        }
        for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            grad_tensor_slice->copy_from(*grad, sample_idx * output_sample_size, output_sample_size);
            math::matmul(const_one, true, a, false, grad_tensor_slice, const_zero, aTg_tensor_slice);
            math::matmul(const_one, false, aTg_tensor_slice, true, c_tensor, const_zero, b_tensor_slice);
            out->copy_from(*b_tensor_slice, 0, input_sample_size, sample_idx * input_sample_size);
        }
#endif
    } else {
        //  out = \sum_{i} (a * b_{i})^T * grad_{i}
        this->b_tensor = b->eval(false);
        if (out == NULL) {
            out = new Tensor<T>(c->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) c] = out;
        }
#if defined(_HAS_CUDA_)
        if (gTa_tensor == nullptr) {
            init_gTa_cublas();
        }
        if (bTaTg_tensor == nullptr) {
            init_bTaTg_cublas();
        }
        //  compute grad^T * (a^T)^T , stored in gTa_tensor in column-major (so in row-major it is a^T * grad)
        this->cublasGemmStridedBatchedWrap(false, true, n_channel_out, n_vert_in, n_vert_out, const_one,
                                           grad->get_ptr(), n_channel_out, output_sample_size, a->get_ptr(), n_vert_in,
                                           0, const_zero, gTa_tensor->get_ptr(), n_channel_out,
                                           n_channel_out * n_vert_in, n_samples);
        //  compute gTa * (b_slice^T)^T , stored in bTaTg_tensor in column-major (so in row-major it is b^T * a^T *
        //  grad)
        this->cublasGemmStridedBatchedWrap(
            false, true, n_channel_out, n_channel_in, n_vert_in, const_one, gTa_tensor->get_ptr(), n_channel_out,
            n_channel_out * n_vert_in, b_tensor->get_ptr(), n_channel_in, input_sample_size, const_zero,
            bTaTg_tensor->get_ptr(), n_channel_out, n_channel_in * n_channel_out, n_samples);
        //  reduced with respect to the sample axis (0-th)
        math::matmul<T>(const_one, false, ones, false, bTaTg_tensor, const_zero, out);
#else
        if (grad_tensor_slice == nullptr) {
            init_grad();
        }
        for (unsigned sample_idx = 0; sample_idx < n_samples; ++input_sample_size) {
            b_tensor_slice->copy_from(*b_tensor, sample_idx * input_sample_size, input_sample_size);
            grad_tensor_slice->copy_from(*grad, sample_idx * output_sample_size, output_sample_size);
            math::matmul(const_one, true, a, false, grad_tensor_slice, const_zero, aTg_tensor_slice);
            math::matmul(const_one, true, b_tensor_slice, false, aTg_tensor_slice,
                         sample_idx == 0 ? const_zero : const_one, out);
        }
#endif
    }
    return out;
}

template class GCNConvOp<int>;
template class GCNConvOp<float>;
template class GCNConvOp<double>;

/**
 * @brief 
 * 
 * @tparam T 
 * @param a 
 * @param b 
 * @param c 
 * @param copy 
 * @param needs_grad 
 * @return GCNConvOp<T>* 
 */
template <typename T>
GCNConvOp<T>* gcnconvop(Tensor<T>* a, Operation<T>* b, Operation<T>* c, bool copy, bool needs_grad) {
    return new GCNConvOp<T>(a, b, c, copy, needs_grad);
}
template GCNConvOp<int>* gcnconvop(Tensor<int>* a, Operation<int>* b, Operation<int>* c, bool copy, bool needs_grad);
template GCNConvOp<float>* gcnconvop(Tensor<float>* a, Operation<float>* b, Operation<float>* c, bool copy,
                                     bool needs_grad);
template GCNConvOp<double>* gcnconvop(Tensor<double>* a, Operation<double>* b, Operation<double>* c, bool copy,
                                      bool needs_grad);

}  // namespace op
}  // namespace magmadnn
