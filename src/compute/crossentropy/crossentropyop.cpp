
#include "compute/crossentropy/crossentropyop.h"

namespace magmadnn {
namespace op {

template <typename T>
CrossEntropyOp<T>::CrossEntropyOp(Operation<T> *x, Operation<T> *y, bool copy, bool needs_grad)
    : Operation<T>::Operation({x, y}, needs_grad), x(x), y(y), copy(copy) {
    /*  x should be (n_samples x n_classes)
        y should be (n_samples x n_classes)
    */
    assert(OP_IS_MATRIX(x));
    assert(OP_IS_MATRIX(y));
    assert(x->get_output_shape(0) == y->get_output_shape(0));
    assert(x->get_output_shape(1) == y->get_output_shape(1));

    this->output_shape = {1};
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "no copy cross entropy not supported yet.\n");
    }
}

template <typename T>
CrossEntropyOp<T>::~CrossEntropyOp() {}

template <typename T>
Tensor<T> *CrossEntropyOp<T>::_eval(bool recompute) {
    x_tensor = x->eval(recompute);
    y_tensor = y->eval(recompute);

    // internal::crossentropy_full(x_tensor, y_tensor, this->softmax, this->output_tensor);
    math::crossentropy(x_tensor, y_tensor, this->output_tensor);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *CrossEntropyOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    this->_grad_cache[(uintptr_t) var] = grad;
    return grad;
}

template class CrossEntropyOp<int>;
template class CrossEntropyOp<float>;
template class CrossEntropyOp<double>;

template <typename T>
Operation<T> *crossentropy(Operation<T> *ground_truth, Operation<T> *predicted, bool copy, bool needs_grad) {
    auto size = ground_truth->get_output_shape(0);
    T norm = static_cast<T>(1.0) / static_cast<T>(size);
    return negative(op::scalarproduct(norm, reducesum(reducesum(product(ground_truth, log(predicted, true)), 1), 0)));
}
template Operation<int> *crossentropy(Operation<int> *, Operation<int> *, bool, bool);
template Operation<float> *crossentropy(Operation<float> *, Operation<float> *, bool, bool);
template Operation<double> *crossentropy(Operation<double> *, Operation<double> *, bool, bool);

}  // namespace op
}  // namespace magmadnn
