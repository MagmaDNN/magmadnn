#pragma once

#include <string>
#include "compute/operation.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"
#include "math/matmul.h"
#include <cublas_v2.h>

//  to do:
//  add support for using sparse matrix input for a
//  use MAGMA for routine?

namespace magmadnn {
namespace op {
/**
 * Compute X^{i}_{jk} = a_{jm} * b^{i}_{mn} * c_{nk} where a is a constant tensor, b, c are opearations
 * 
 * @tparam T 
 * @param a Tensor pointer to the constant tensor
 */
template <typename T>
class GCNConvOp : public Operation<T> {
   public:
    GCNConvOp(Tensor<T> *a, Operation<T> *b, Operation<T> *c, bool copy = true, bool needs_grad = true);
    ~GCNConvOp(void);
    inline std::string to_string(void) {
        return "GCNCONV( Tensor({" + std::to_string(a->get_shape(0)) + ", " + std::to_string(a->get_shape(1)) + "}) * " + b->to_string() + " * " + c->to_string() + " )";
    }

   private:

#if defined(_HAS_CUDA_)
    static void cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, T alpha, T *A, int lda,
                                             long long strideA, T *B, int ldb, long long strideB, T beta, T *C, int ldc,
                                             long long strideC, int batch);
#endif
    inline static void delete_if_not_nullptr(Tensor<T> *ptr) {
        if (ptr != nullptr) {
            delete ptr;
            ptr = nullptr;
        }
    }
    const T const_one = (T) 1;
    const T const_zero = (T) 0;

   protected:
    unsigned n_samples;
    unsigned n_vert_in, n_vert_out;
    unsigned n_channel_in, n_channel_out;
    unsigned input_sample_size, output_sample_size;
    Tensor<T> *a;
    Operation<T> *b;
    Operation<T> *c;
    bool copy;

    Tensor<T> *b_tensor;
    Tensor<T> *b_tensor_slice;
    Tensor<T> *c_tensor;

    //  for native
    Tensor<T> *ab_tensor_slice;
    Tensor<T> *abc_tensor_slice;
    Tensor<T> *grad_tensor_slice;
    Tensor<T> *aTg_tensor_slice;

    //  for cublas
    Tensor<T> *cTbT_tensor;
    Tensor<T> *gTa_tensor;
    Tensor<T> *bTaTg_tensor;
    Tensor<T> *ones;  //  for reducing

#if defined(_HAS_CUDA_)
    void init_eval_cublas(void);
    void init_gTa_cublas(void);
    void init_bTaTg_cublas(void);
#else
    void init_eval(void);
    void init_grad(void);
#endif

    Tensor<T> *_eval(bool recompute);
    Tensor<T> *_grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad);
};

template <typename T>
GCNConvOp<T> *gcnconvop(Tensor<T> *a, Operation<T> *b, Operation<T> *c, bool copy = true, bool needs_grad = true);

}  // namespace op
}  // namespace magmadnn
