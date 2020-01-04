#include "magmadnn/config.h"
#include "compute/reducesum/reducesumop.h"

namespace magmadnn {
namespace op {

template <typename T>
ReduceSumOp<T>::ReduceSumOp(Operation<T> *x, int axis, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), axis(axis), copy(copy) {
    std::vector<unsigned int> const &x_output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    /* don't allow an axis greater than size of shape */
    assert(axis < (int) x_output_shape.size());

    if (x_output_shape.size() == 1 || axis == -1) {
        /* x is a 1D vector. simply sum elements */
        this->output_shape = {1};
    } else if (x_output_shape.size() == 2) {
        /* matrix reduction */
        if (axis == 0) {
            /* the number of cols */
            this->output_shape = {1, x_output_shape.at(1)};
            ones = new Tensor<T>({x_output_shape.at(0)}, {ONE, {}}, this->mem_type);
        } else {
            /* the number of rows */
            this->output_shape = {x_output_shape.at(0), 1};
            ones = new Tensor<T>({x_output_shape.at(1)}, {ONE, {}}, this->mem_type);
        }
    } else if (this->mem_type == HOST) {
        this->output_shape = x_output_shape;
        std::fprintf(stderr, "ReduceSum not available for general tensors with more than 2 axes.\n");
    }

    if (copy) {
        /* init to ones */
        this->output_tensor = new Tensor<T>(this->get_output_shape(), {ONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "Non-Copy ReduceSum not supported.\n");
    }

#if defined(MAGMADNN_HAVE_CUDA)

    /* create a temporary descriptor for x, since we do not have its tensor yet (it's an operation) and
        therefore cannot call get_cudnn_tensor_descriptor(). This allows us to get the
        workspace size from CuDNN here in the constructor, rather than in eval. */
    cudnnTensorDescriptor_t x_tmp_descriptor;
    int x_n = 1, x_c = 1, x_h = 1, x_w = 1;
    unsigned int x_axes = x->get_output_shape().size();

    if (x_axes > 4 || x_axes == 0) {
        fprintf(stderr, "Unsupported operation.\n");
    }
    if (x_axes == 4) {
        x_w = x->get_output_shape(3);
    }
    if (x_axes >= 3) {
        x_h = x->get_output_shape(2);
    }
    if (x_axes >= 2) {
        x_c = x->get_output_shape(1);
    }
    if (x_axes >= 1) {
        x_n = x->get_output_shape(0);
    }

    cudnnErrchk(cudnnCreateTensorDescriptor(&x_tmp_descriptor));
    cudnnErrchk(cudnnSetTensor4dDescriptor(x_tmp_descriptor, CUDNN_TENSOR_NCHW,
                                           ::magmadnn::internal::get_cudnn_data_type((T) 0), x_n, x_c, x_h, x_w));

    cudnnErrchk(cudnnCreateReduceTensorDescriptor(&reduce_settings.descriptor));
    cudnnErrchk(cudnnSetReduceTensorDescriptor(
        reduce_settings.descriptor, CUDNN_REDUCE_TENSOR_ADD, ::magmadnn::internal::get_cudnn_data_type((T) 0),
        CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
    cudnnErrchk(cudnnGetReductionWorkspaceSize(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cudnn_handle, reduce_settings.descriptor, x_tmp_descriptor,
        this->output_tensor->get_cudnn_tensor_descriptor(), &reduce_settings.workspace_size));
    cudaErrchk(cudaMalloc((void **) &reduce_settings.workspace, reduce_settings.workspace_size * sizeof(T)));

    cudnnErrchk(cudnnDestroyTensorDescriptor(x_tmp_descriptor));
#endif
}

template <typename T>
ReduceSumOp<T>::~ReduceSumOp() {
    if (ones != NULL) delete ones;

#if defined(MAGMADNN_HAVE_CUDA)
    cudnnErrchk(cudnnDestroyReduceTensorDescriptor(reduce_settings.descriptor));
    cudaErrchk(cudaFree(reduce_settings.workspace));
#endif
}

template <typename T>
Tensor<T> *ReduceSumOp<T>::_eval(bool recompute) {

   x_tensor = x->eval(recompute);

   if (!copy) {
      std::fprintf(stderr, "Non-Copy ReduceSum not supported.\n");
      return this->output_tensor;
   }

   if (this->mem_type == HOST) {
      math::reduce_sum(x_tensor, axis, ones, this->output_tensor);
   }
#if defined(MAGMADNN_HAVE_CUDA)
   else {
      this->reduce_settings.cudnn_handle = this->get_cudnn_handle();
      math::reduce_sum_device(x_tensor, axis, this->output_tensor, this->reduce_settings);
      // Assume cudnn_handle is associated with custream
      cudaStreamSynchronize(this->get_custream());
   }
#endif
   return this->output_tensor;
}

template <typename T>
Tensor<T> *ReduceSumOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* repeat grad along specified axis */

    /* output_shape = x.shape */
    /* output_shape[axis] = 1 */
    /* tile_scaling = x.shape // output_shape */
    /* reshape grad to output_shape */
    /* tile grad tile_scaling */

    Tensor<T> *out = this->_grad_cache[(uintptr_t) var];

    if (out == NULL) {
        /* the gradient output will have the same shape as this operations input */
        out = new Tensor<T>(x->get_output_shape(), {NONE, {}}, this->mem_type);
#if defined(MAGMADNN_HAVE_CUDA)
        out->set_custream(this->get_custream());
        out->set_cublas_handle(this->get_cublas_handle());
#endif
        this->_grad_cache[(uintptr_t) var] = out;
    }

    internal::reduce_sum_grad(grad, this->axis, out);
#if defined(MAGMADNN_HAVE_CUDA)
    cudaStreamSynchronize(this->get_custream());
#endif

    return out;
}

template class ReduceSumOp<int>;
template class ReduceSumOp<float>;
template class ReduceSumOp<double>;

template <typename T>
ReduceSumOp<T> *reducesum(Operation<T> *x, int axis, bool copy, bool needs_grad) {
    return new ReduceSumOp<T>(x, axis, copy, needs_grad);
}
template ReduceSumOp<int> *reducesum(Operation<int> *x, int axis, bool copy, bool needs_grad);
template ReduceSumOp<float> *reducesum(Operation<float> *x, int axis, bool copy, bool needs_grad);
template ReduceSumOp<double> *reducesum(Operation<double> *x, int axis, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
