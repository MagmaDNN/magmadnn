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

template <typename T>
class variable : public operation<T> {
public:
    variable (std::string name, tensor<T> *val) : operation<T>::operation(), name(name), val(val) {}

    tensor<T>* eval();

protected:
    std::string name;
    tensor<T> *val;
};

} // namespace skepsi