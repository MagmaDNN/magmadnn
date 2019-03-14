/**
 * @file variable.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/variable.h"

namespace skepsi {
namespace op {

template <typename T>
Tensor<T>* Variable<T>::eval() {
    return val;
}

// compile for int, float, double
template class Variable<int>;
template class Variable<float>;
template class Variable<double>;

template <typename T>
Variable<T>* var(std::string name, Tensor<T>* val) {
    return new Variable<T> (name, val);
}
template Variable<int>* var(std::string name, Tensor<int>* val);
template Variable<float>* var(std::string name, Tensor<float>* val);
template Variable<double>* var(std::string name, Tensor<double>* val);

} // namespace op
} // namespace skepsi