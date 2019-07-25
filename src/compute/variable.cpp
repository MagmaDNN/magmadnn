/**
 * @file variable.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-18
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/variable.h"

namespace magmadnn {
namespace op {

Variable::Variable(std::string name, const std::vector<index_t>& shape, DataType dtype, tensor_filler_t filler,
                   memory_t mem_type)
    : Operation::Operation(), name_(name) {
    this->output_tensor_ = Tensor(shape, dtype, filler, mem_type);

    this->output_shape_ = shape;
    this->mem_type_ = mem_type;
    this->dtype_ = dtype;

    this->has_been_computed_ = true;
}

Variable::Variable(std::string name, Tensor& val) : Operation::Operation(), name_(name) {
    this->output_tensor_ = val;

    use_tensor_settings(val);

    this->has_been_computed_ = true;
}

Tensor& Variable::_eval(bool recompute) { return this->output_tensor_; }

Tensor& Variable::_grad(Operation* consumer, Operation* var, const Tensor& grad) {
    /* TODO -- if (var == this) return 1; */

    /* TODO -- find a better way to do this without const_cast */
    return const_cast<Tensor&>(grad);
}

}  // namespace op
}  // namespace magmadnn