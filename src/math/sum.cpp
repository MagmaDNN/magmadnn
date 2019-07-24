/**
 * @file sum.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-10
 *
 * @copyright Copyright (c) 2019
 */
#include "math/sum.h"

namespace magmadnn {
namespace math {

template <typename T>
void sum(const std::vector<std::reference_wrapper<const Tensor>>& tensors, Tensor& out) {
    /* early exit */
    if (tensors.size() == 0) return;

    MAGMADNN_ASSERT(::magmadnn::utilities::do_tensors_match(GetDataType<T>::value,
                                                            tensors.front().get().get_memory_type(), tensors),
                    "tensors memory types and data types must match");

    if (out.get_memory_type() == HOST) {
        T* out_ptr = out.get_ptr<T>();
        size_t size = out.size();

        /* iterate over tensors first for cache efficiency */
        bool first = true;
        for (const auto& t : tensors) {
            const T* t_ptr = t.get().get_ptr<T>();

            if (first) {
                /* assign if we're the first element */
                for (index_t i = 0; i < size; i++) {
                    out_ptr[i] = t_ptr[i];
                }
                first = false;
            } else {
                /* continue to accumulate after first element */
                for (index_t i = 0; i < size; i++) {
                    out_ptr[i] += t_ptr[i];
                }
            }
        }
    }
#if defined(_HAS_CUDA_)
    else {
        sum_device<T>(tensors, out);
    }
#endif
}
#define COMPILE_SUM(type) template void sum<type>(const std::vector<std::reference_wrapper<const Tensor>>&, Tensor&);
CALL_FOR_ALL_TYPES(COMPILE_SUM)
#undef COMPILE_SUM

}  // namespace math
}  // namespace magmadnn