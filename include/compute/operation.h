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
    /** The operation class serves as an abstract object, which all tensors operations descend
     *  from. It is used to build a computation tree.
     */
    operation() {}
    operation(std::vector<operation<T>*> children) : children(children) {}
	virtual ~operation() {
        for (unsigned int i = 0; i < children.size(); i++)
            delete children[i];
    }

    /** Returns the operation's evaluated tensor.
     * @return tensor<T>* 
     */
    virtual tensor<T>* eval() = 0;

    /** string form of the given operation. Expands on children.
     * @return std::string 
     */
    virtual std::string to_string() = 0;
    
protected:
    std::vector<operation<T>*> children;
};

} // namespace op
} // namespace skepsi
