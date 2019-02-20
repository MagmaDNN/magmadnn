/**
 * @file add_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/add_op.h"

namespace skepsi {
namespace op {

template <typename T>
tensor<T>* add_op<T>::eval() {
	tensor<T>* a_tensor = a->eval();
	tensor<T>* b_tensor = b->eval();

	for (unsigned int i = 0; i < a_tensor->get_size(); i++) {
		b_tensor->set(i, a_tensor->get(i) + b_tensor->get(i));
	}
	return b_tensor;
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