/**
 * @file sumop.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/sum/sumop.h"

namespace magmadnn {
namespace op {

SumOp::SumOp(std::vector<Operation *> ops) : Operation<T>::Operation(ops), ops(ops) {
    if (ops.empty()) {
        return;
    }

    std::vector<Operation *>::const_iterator it = ops.begin();
    unsigned int first_size = (*it)->get_output_size();
    for (it++; it != ops.end(); it++) {
        assert((*it)->get_output_size() == first_size);
    }

    this->output_shape = ops.at(0)->get_output_shape();
    this->mem_type = ops.at(0)->get_memory_type();

    this->output_tensor =
        Tensor(ops.at(0)->get_output_shape(), ops.at(0).dtype(), {ZERO, {}}, ops.at(0)->get_memory_type());
}

Tensor &SumOp::_eval(bool recompute) {
    std::vector<Tensor *> vals(ops.size());

    for (unsigned int i = 0; i < ops.size(); i++) {
        vals[i] = ops[i]->eval(recompute);
    }

    internal::sum_full(vals, this->output_tensor);

    return this->output_tensor;
}

Tensor &SumOp::_grad(Operation *consumer, Operation *var, Tensor &grad) { return grad; }

std::string SumOp::to_string() override {
    std::string ret = "(";
    for (typename std::vector<Operation *>::iterator vit = this->ops.begin(); vit != this->ops.end(); vit++) {
        if (vit != ops.begin()) {
            ret += "+";
        }
        ret += " " + (*vit)->to_string() + " ";
    }
    return ret + ")";
}

}  // namespace op
}  // namespace magmadnn