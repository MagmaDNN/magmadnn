/**
 * @file sum_internal.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/sum/sum_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void sum_full(std::vector<Tensor<T> *> &vals, Tensor<T> &out) {

    if (vals.at(0)->get_memory_type() == HOST) {

        T sum;
        for (unsigned int idx = 0; idx < vals[0]->get_size(); idx++) {
            sum = (T) 0;
            for (unsigned int i = 0; i < vals.size(); i++) {
                sum += (vals[i])->get(idx);
            }
            out.set(idx, sum);
        }
    }
    #if defined(_HAS_CUDA_)
    else {
        sum_full_device(vals, out);
    }
    #endif
}
template void sum_full(std::vector<Tensor<int> *> &vals, Tensor<int> &out);
template void sum_full(std::vector<Tensor<float> *> &vals, Tensor<float> &out);
template void sum_full(std::vector<Tensor<double> *> &vals, Tensor<double> &out);

}   // namespace internal
}   // namespace magmadnn