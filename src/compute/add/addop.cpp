/**
 * @file add_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/add/addop.h"

namespace magmadnn {
namespace op {

template <typename T>
AddOp<T>::AddOp(Operation<T>* a, Operation<T>* b, bool copy, bool needs_grad) : 
	Operation<T>::Operation({a,b}, needs_grad), a(a), b(b), copy(copy) {
	
	assert( a->get_memory_type() == b->get_memory_type() );
	assert( a->get_output_size() == b->get_output_size() || a->get_output_size() == 1 || b->get_output_size() == 1 );

	/* if a is scalar then use b's size */
	if (a->get_output_size() == 1) {
		this->output_shape = b->get_output_shape();
	} else {
		/* other wise a's size is good */
		this->output_shape = a->get_output_shape();
	}
	this->mem_type = a->get_memory_type();

	/* Go ahead and create copy tensor if we can */
	if (copy) {
		this->ret = new Tensor<T> (this->output_shape, this->mem_type);
	}
}

template <typename T>
Tensor<T>* AddOp<T>::eval() {
	a_tensor = a->eval();
	b_tensor = b->eval();

	if (!copy) this->ret = b_tensor;

	if (a_tensor->get_size() == 1) {
		internal::tensor_scalar_add_full(a_tensor->get(0), b_tensor, this->ret);
	} else if (b_tensor->get_size() == 1) {
		internal::tensor_scalar_add_full(b_tensor->get(0), a_tensor, this->ret);
	} else {
		internal::geadd_full((T)1, a_tensor, (T)1, b_tensor, this->ret);
	}
	
	return this->ret;
} 

template <typename T>
Operation<T> *AddOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
	return grad;
}
template class AddOp<int>;
template class AddOp<float>;
template class AddOp<double>;


template <typename T>
AddOp<T>* add(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) {
    return new AddOp<T> (a, b, copy, needs_grad);
}
template AddOp<int>* add(Operation<int> *a, Operation<int> *b, bool copy, bool needs_grad);
template AddOp<float>* add(Operation<float> *a, Operation<float> *b, bool copy, bool needs_grad);
template AddOp<double>* add(Operation<double> *a, Operation<double> *b, bool copy, bool needs_grad);

} // namespace op
} // namespace magmadnn