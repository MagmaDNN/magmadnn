/**
 * @file operation.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "tensor/tensor.h"

namespace skepsi {

template <typename T>
class operation {
public: 
    operation() {}
    operation(std::vector<operation<T>*> children) : children(children) {}
	virtual ~operation() {}

    virtual tensor<T>* eval() = 0;
    
protected:
    std::vector<operation<T>*> children;
};

} // namespace skepsi
