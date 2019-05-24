
#include "compute/div/divop.h"

namespace magmadnn {
namespace op {

template <typename T>
DivOp<T>::DivOp(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) 
    : Operation<T>::Operation({a,b}, needs_grad), a(a), b(b), copy(copy) {
    
    this->mem_type = a->get_memory_type();

    unsigned int a_size = a->get_output_size();
    unsigned int b_size = b->get_output_size();

    if (a_size == 1 && b_size == 1) {
        /* scalar-scalar ,but we can use TENSOR_DIV_TENSOR */
        op_type = internal::TENSOR_DIV_TENSOR;
        this->output_shape = a->get_output_shape();
    } else if (a_size == 1) {
        /* scalar-tensor */
        op_type = internal::SCALAR_DIV_TENSOR;
        this->output_shape = b->get_output_shape();
    } else if (b_size == 1) {
        /* tensor-scalar */
        op_type = internal::TENSOR_DIV_SCALAR;
        this->output_shape = a->get_output_shape();
    } else {
        /* tensor-tensor for now (TODO : include VEC cases) */
        assert( a_size == b_size );
        op_type = internal::TENSOR_DIV_TENSOR;
        this->output_shape = a->get_output_shape();
    }

    if (copy) {
        this->ret = new Tensor<T> (this->output_shape, {NONE,{}}, this->mem_type);
    }
}

template <typename T>
Tensor<T> *DivOp<T>::eval() {
    a_tensor = a->eval();
    b_tensor = b->eval();

    if (!copy) this->ret = b_tensor;

    switch (op_type) {
        case internal::TENSOR_DIV_TENSOR:
            internal::tensor_div_tensor_full(a_tensor, b_tensor, this->ret); break;
        case internal::SCALAR_DIV_TENSOR:
            a_tensor->get_memory_manager()->sync(true);
            internal::scalar_div_tensor_full(a_tensor->get(0), b_tensor, this->ret); break;
        case internal::TENSOR_DIV_SCALAR:
            b_tensor->get_memory_manager()->sync(true);
            internal::tensor_div_scalar_full(a_tensor, b_tensor->get(0), this->ret); break;
        default:
            std::fprintf(stderr, "This type of div is not yet supported.\n");
    }

    return this->ret;
}

template <typename T>
Operation<T> *DivOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
    /* grad is ... */
    return grad;
}

template class DivOp<int>;
template class DivOp<float>;
template class DivOp<double>;


template <typename T>
DivOp<T> *div(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) {
    return new DivOp<T>(a, b, copy, needs_grad);
}
template DivOp<int> *div(Operation<int>*, Operation<int>*, bool, bool);
template DivOp<float> *div(Operation<float>*, Operation<float>*, bool, bool);
template DivOp<double> *div(Operation<double>*, Operation<double>*, bool, bool);


}   // namespace op
}   // namespace magmadnn
