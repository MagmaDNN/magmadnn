/**
 * @file gradientdescent_internal.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#include "optimizer/gradientdescent/gradientdescent_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
magmadnn_error_t gradientdescent_update_internal(Tensor<T> *var, Tensor<T> *grad, T learning_rate) {
    magmadnn_error_t err = (magmadnn_error_t) 2;

    if (var->get_memory_type() == HOST) {
        T *var_ptr = var->get_ptr();
        T *grad_ptr = grad->get_ptr();
        unsigned int size = var->get_size();

        for (unsigned int i = 0; i < size; i++) {
            var_ptr[i] -= learning_rate * grad_ptr[i];
        }
        err = (magmadnn_error_t) 0;
    }
#if defined(_HAS_CUDA_)
    else {
        err = gradientdescent_update_internal_device(var, grad, learning_rate);
    }
#endif

    return (magmadnn_error_t) err;
}
template magmadnn_error_t gradientdescent_update_internal(Tensor<int> *var, Tensor<int> *grad, int learning_rate);
template magmadnn_error_t gradientdescent_update_internal(Tensor<float> *var, Tensor<float> *grad, float learning_rate);
template magmadnn_error_t gradientdescent_update_internal(Tensor<double> *var, Tensor<double> *grad,
                                                          double learning_rate);

}  // namespace internal
}  // namespace magmadnn