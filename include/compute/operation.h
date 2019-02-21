/**
 * @file operation.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include "tensor/tensor.h"

namespace skepsi {
namespace op {

template <typename T>
class operation {
public: 
    operation() {}
    operation(std::vector<operation<T>*> children) : children(children) {}
	virtual ~operation() {
        for (unsigned int i = 0; i < children.size(); i++)
            delete children[i];
    }

    virtual tensor<T>* eval() = 0;

    virtual std::string to_string() = 0;
    
protected:
    std::vector<operation<T>*> children;
};

} // namespace op
} // namespace skepsi
