
#include "compute/log/logop.h"

namespace magmadnn {
namespace op {

template <typename T>
LogOp<T>::LogOp(Operation<T> *x, bool copy, bool needs_grad) 
    : Operation<T>::Operation({x}, needs_grad), x(x), copy(copy) {
    
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE,{}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *LogOp<T>::_eval(bool recompute) {

    x_tensor = x->eval(recompute);

    if (!copy) this->output_tensor = x_tensor;

    internal::log_full(x_tensor, this->output_tensor);

    return this->output_tensor;
}

template <typename T>
Operation<T> *LogOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    /* TODO : grad * (1/x) */
    return product(grad, div((Operation<T> *)scalar("1", 1, this->mem_type), x, false, false), false);
}

template class LogOp<int>;
template class LogOp<float>;
template class LogOp<double>;


template <typename T>
LogOp<T> *log(Operation<T> *x, bool copy, bool needs_grad) {
    return new LogOp<T> (x, copy, needs_grad);
}
template LogOp<int> *log(Operation<int> *x, bool copy, bool needs_grad);
template LogOp<float> *log(Operation<float> *x, bool copy, bool needs_grad);
template LogOp<double> *log(Operation<double> *x, bool copy, bool needs_grad);


}   // namespace op
}   // namespace magmadnn
