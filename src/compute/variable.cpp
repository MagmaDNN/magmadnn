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

} // namespace op
} // namespace skepsi