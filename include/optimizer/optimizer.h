/**
 * @file optimizer.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-29
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <string>
#include <vector>
#include "compute/operation.h"

namespace magmadnn {
namespace optimizer {

template <typename T>
class Optimizer {
public:
    Optimizer() {}

    virtual ~Optimizer() {}


    virtual void minimize(op::Operation<T> *obj_func, const std::vector<op::Operation<T> *>& wrt) = 0;

    virtual std::string get_name() { return _name; }

protected:
    virtual void update(op::Operation<T> *var, Tensor<T> *grad) = 0;

    op::Operation<T> *_obj_func;
    std::string _name = "Generic Optimizer";
};

}   // namespace optimizer
}   // namespace magmadnn