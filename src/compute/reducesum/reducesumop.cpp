
#include "compute/reducesum/reducesumop.h"

namespace magmadnn {
namespace op {

template <typename T>
ReduceSumOp<T>::ReduceSumOp() : Operation<T>::Operation() {
    /* setup code in here */
    this->output_shape = /* ... */
    this->mem_type = /* ... */
}

template <typename T>
Tensor<T> *ReduceSumOp<T>::eval() {
    /* eval code in here ... */
    return ret;
}

template <typename T>
Operation<T> *ReduceSumOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    /* return gradient in here ... */
    return grad;
}

template class ReduceSumOp<int>;
template class ReduceSumOp<float>;
template class ReduceSumOp<double>;


template <typename T>
ReduceSumOp<T> *reducesum() {
    return new ReduceSumOp<T>();
}
template ReduceSumOp<int> *reducesum();
template ReduceSumOp<float> *reducesum();
template ReduceSumOp<double> *reducesum();


}   // namespace op
}   // namespace magmadnn