/**
 * @file add_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/add/add_op.h"

namespace skepsi {
namespace op {

template <typename T>
add_op<T>::add_op(operation<T>* a, operation<T>* b, bool copy) : 
	operation<T>::operation({a,b}), a(a), b(b), copy(copy) {
	
	assert( a->get_memory_type() == b->get_memory_type() );
	assert( a->get_output_size() == b->get_output_size() );

	this->output_shape = a->get_output_shape();
	this->mem_type = a->get_memory_type();

	/* Go ahead and create copy tensor if we can */
	if (copy)
		ret = new tensor<T> (this->output_shape, this->mem_type);
}

template <typename T>
tensor<T>* add_op<T>::eval() {
	a_tensor = a->eval();
	b_tensor = b->eval();

	if (!copy) ret = b_tensor;

	internal::geadd_full((T)1, a_tensor, (T)1, b_tensor, ret);
	
	return ret;
} 
template class add_op<int>;
template class add_op<float>;
template class add_op<double>;


template <typename T>
add_op<T>* add(operation<T> *a, operation<T> *b, bool copy) {
    return new add_op<T> (a, b, copy);
}
template add_op<int>* add(operation<int> *a, operation<int> *b, bool copy);
template add_op<float>* add(operation<float> *a, operation<float> *b, bool copy);
template add_op<double>* add(operation<double> *a, operation<double> *b, bool copy);

} // namespace op
} // namespace skepsi