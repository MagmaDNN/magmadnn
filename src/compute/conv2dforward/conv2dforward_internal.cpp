
#include "compute/conv2dforward/conv2dforward_internal.h"

namespace magmadnn {
namespace internal {

template <typename T>
void conv2dforward_full(Tensor<T> *in, Tensor<T> *out) {
    /*
    
     */
}
template void conv2dforward_full(Tensor<int> *in, Tensor<int> *out);
template void conv2dforward_full(Tensor<float> *in, Tensor<float> *out);
template void conv2dforward_full(Tensor<double> *in, Tensor<double> *out);

}   // namespace op
}   // namespace magmadnn