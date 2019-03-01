/**
 * @file variable.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include <tensor/tensor.h>
#include "operation.h"

namespace skepsi {
namespace op {

/** Variable Operation. The most basic operation; it simply wraps around a tensor.
 * @tparam T 
 */
template <typename T>
class variable : public operation<T> {
public:
    variable (std::string name, tensor<T> *val) : operation<T>::operation(), name(name), val(val) {}

    tensor<T>* eval();

    std::string to_string() { return name; }

protected:
    std::string name;
    tensor<T> *val;
};

/** Returns a new variable operation. The variable wraps around val with name name.
 * @tparam T 
 * @param name the name of the variable. This does not effect computation. It will allow to_string() methods to work, however.
 * @param val tensor to wrap around
 * @return variable<T>* 
 */
template <typename T>
variable<T>* var(std::string name, tensor<T>* val);

} // namespace op
} // namespace skepsi
