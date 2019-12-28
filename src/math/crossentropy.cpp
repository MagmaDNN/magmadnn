/**
 * @file crossentropy.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-12
 *
 * @copyright Copyright (c) 2019
 */
#include "math/crossentropy.h"

#include <cassert>

namespace magmadnn {
namespace math {

template <typename T>
void crossentropy(Tensor<T> *predicted, Tensor<T> *ground_truth, Tensor<T> *out) {
    assert(T_IS_SCALAR(out) && T_IS_MATRIX(predicted) && T_IS_MATRIX(ground_truth));

    if (out->get_memory_type() == HOST) {
        T *predicted_ptr = predicted->get_ptr();
        T *ground_truth_ptr = ground_truth->get_ptr();
        T *out_ptr = out->get_ptr();
        unsigned int n_samples = predicted->get_shape(0);
        unsigned int n_classes = predicted->get_shape(1);
        T sum = (T) 0;

        /* compute dot product : ground_truth[i,:].log(predicted[i,:]) */
        for (unsigned int i = 0; i < n_samples * n_classes; i++) {
            /* TODO -- investigate "if(ground_truth[.] == 0) continue;" would speed this up.
                        It might due to no log call, but introducing a conditional branch might hurt the pipeline. */

            if (predicted_ptr[i] <= 0) continue; /* avoids NaN from log */
            sum += ground_truth_ptr[i] * (T) log(predicted_ptr[i]);
        }

        out_ptr[0] = -sum / ((T) n_samples);
    }
#if defined(_HAS_CUDA_)
    else {
        crossentropy_device(predicted, ground_truth, out);
    }
#endif
}
template void crossentropy(Tensor<int> *predicted, Tensor<int> *ground_truth, Tensor<int> *out);
template void crossentropy(Tensor<float> *predicted, Tensor<float> *ground_truth, Tensor<float> *out);
template void crossentropy(Tensor<double> *predicted, Tensor<double> *ground_truth, Tensor<double> *out);

}  // namespace math
}  // namespace magmadnn
