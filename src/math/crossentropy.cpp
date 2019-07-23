/**
 * @file crossentropy.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-12
 *
 * @copyright Copyright (c) 2019
 */
#include "math/crossentropy.h"

namespace magmadnn {
namespace math {

template <typename T>
void crossentropy(const Tensor &predicted, const Tensor &ground_truth, Tensor &out) {
    /* TODO -- these macros expect tensor pointers */
    // assert(T_IS_SCALAR(out) && T_IS_MATRIX(predicted) && T_IS_MATRIX(ground_truth));
    MAGMADNN_ASSERT(
        TYPES_MATCH(T, predicted.dtype()) && TYPES_MATCH(T, ground_truth.dtype()) && TYPES_MATCH(T, out.dtype()),
        "template type must match tensor type");

    if (out.get_memory_type() == HOST) {
        const T *predicted_ptr = predicted.get_ptr<T>();
        const T *ground_truth_ptr = ground_truth.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();
        index_t n_samples = predicted.shape(0);
        index_t n_classes = predicted.shape(1);
        T sum = static_cast<T>(0);

        /* compute dot product : ground_truth[i,:].log(predicted[i,:]) */
        for (index_t i = 0; i < n_samples * n_classes; i++) {
            /* TODO -- investigate "if(ground_truth[.] == 0) continue;" would speed this up.
                        It might due to no log call, but introducing a conditional branch might hurt the pipeline. */

            if (predicted_ptr[i] <= 0) continue; /* avoids NaN from log */
            sum += ground_truth_ptr[i] * static_cast<T>(log(predicted_ptr[i]));
        }

        out_ptr[0] = -sum / static_cast<T>(n_samples);
    }
#if defined(_HAS_CUDA_)
    else {
        crossentropy_device<T>(predicted, ground_truth, out);
    }
#endif
}
#define COMPILE_CROSSENTROPY(type) template void crossentropy<type>(const Tensor &, const Tensor &, Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_CROSSENTROPY)
#undef COMPILE_CROSSENTROPY

}  // namespace math
}  // namespace magmadnn