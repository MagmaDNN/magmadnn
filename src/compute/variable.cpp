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
tensor<T>* variable<T>::eval() {
    return val;
}

// compile for int, float, double
template class variable<int>;
template class variable<float>;
template class variable<double>;

template <typename T>
variable<T>* var(std::string name, tensor<T>* val) {
    return new variable<T> (name, val);
}
template variable<int>* var(std::string name, tensor<int>* val);
template variable<float>* var(std::string name, tensor<float>* val);
template variable<double>* var(std::string name, tensor<double>* val);

} // namespace op
} // namespace skepsi