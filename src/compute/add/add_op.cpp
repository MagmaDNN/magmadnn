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
tensor<T>* add_op<T>::eval() {
	tensor<T> *a_tensor = a->eval();
	tensor<T> *b_tensor = b->eval();

	tensor<T> *c_tensor;
	if (copy) 
		c_tensor = new tensor<T> (a_tensor->get_shape(), a_tensor->get_memory_type());
	else
		c_tensor = b_tensor;

	internal::geadd_full((T)1, a_tensor, (T)1, b_tensor, c_tensor);
	
	return c_tensor;
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