/**
 * @file sum.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-10
 *
 * @copyright Copyright (c) 2019
 */
#include "math/sum.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void sum(const std::vector<Tensor<T>*>& tensors, Tensor<T>* out) {
    /* early exit */
    if (tensors.size() == 0) return;

    /* iterate over all the tensors and ensure the same memory type */
    for (const auto& t : tensors) {
        assert(T_IS_SAME_MEMORY_TYPE(out, t));
    }

    if (out->get_memory_type() == HOST) {
        T* out_ptr = out->get_ptr();
        T* t_ptr;
        unsigned int size = out->get_size();

        /* iterate over tensors first for cache efficiency */
        for (typename std::vector<Tensor<T>*>::const_iterator it = tensors.begin(); it != tensors.end(); it++) {
            t_ptr = (*it)->get_ptr(); /* get the pointer to this tensors memory */

            if (it == tensors.begin()) {
                /* assign if we're the first element */
                for (unsigned int i = 0; i < size; i++) {
                    out_ptr[i] = t_ptr[i];
                }
            } else {
                /* continue to accumulate after first element */
                for (unsigned int i = 0; i < size; i++) {
                    out_ptr[i] += t_ptr[i];
                }
            }
        }
    }
#if defined(_HAS_CUDA_)
    else {
        sum_device(tensors, out);
    }
#endif
}
template void sum(const std::vector<Tensor<int>*>& tensors, Tensor<int>* out);
template void sum(const std::vector<Tensor<float>*>& tensors, Tensor<float>* out);
template void sum(const std::vector<Tensor<double>*>& tensors, Tensor<double>* out);

}  // namespace math
}  // namespace magmadnn
