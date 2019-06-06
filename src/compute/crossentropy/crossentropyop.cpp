
#include "compute/crossentropy/crossentropyop.h"

namespace magmadnn {
namespace op {

template <typename T>
CrossEntropyOp<T>::CrossEntropyOp(Operation<T> *x, Operation<T> *y, bool copy, bool needs_grad)
: Operation<T>::Operation({x,y}, needs_grad), x(x), y(y), copy(copy) {


    /*  x should be (n_samples x n_classes)
        y should be (n_samples x n_classes)
    */
    assert( OP_IS_MATRIX(x) );
    assert( OP_IS_MATRIX(y) );
    assert( x->get_output_shape(0) == y->get_output_shape(0) );
    assert( x->get_output_shape(1) == y->get_output_shape(1) );
    
    this->output_shape = {1};
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T> (this->output_shape, {NONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "no copy cross entropy not supported yet.\n");
    }

    softmax = new Tensor<T> (x->get_output_shape(), {NONE, {}}, this->mem_type);
}

template <typename T>
CrossEntropyOp<T>::~CrossEntropyOp() {
    delete this->softmax;
}

template <typename T>
Tensor<T> *CrossEntropyOp<T>::_eval(bool recompute) {

    x_tensor = x->eval(recompute);
    y_tensor = y->eval(recompute);

    internal::crossentropy_full(x_tensor, y_tensor, this->softmax, this->output_tensor);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *CrossEntropyOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* softmax(x) -= 1 and /= N */
    return grad;
}

template class CrossEntropyOp<int>;
template class CrossEntropyOp<float>;
template class CrossEntropyOp<double>;


template <typename T>
CrossEntropyOp<T> *crossentropy(Operation<T> *x, Operation<T> *y, bool copy, bool needs_grad) {
    return new CrossEntropyOp<T>(x, y, copy, needs_grad);
}
template CrossEntropyOp<int> *crossentropy(Operation<int>*, Operation<int>*, bool, bool);
template CrossEntropyOp<float> *crossentropy(Operation<float>*, Operation<float>*, bool, bool);
template CrossEntropyOp<double> *crossentropy(Operation<double>*, Operation<double>*, bool, bool);


}   // namespace op
}   // namespace magmadnn