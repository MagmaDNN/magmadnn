/**
 * @file unaryop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-08-09
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "compute/compute_graph.h"
#include "compute/operation.h"

namespace magmadnn {
namespace op {

template <typename UnaryOpType>
class UnaryOp : public Operation {
   public:
    UnaryOp(Operation *x);

    std::string to_string() const override { return "UNARY_OP(" + x->to_string() + ")"; }

   protected:
    Tensor &_eval(bool recompute = true) override;
    Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) override;

    Operation *x;
};

#define MAKE_UNARY(name)                                                                   \
    inline Operation *name(Operation *a, Operation *out) {                                 \
        return default_graph.add_operation<UnaryOp<::magmadnn::math::name##_map>>(a, out); \
    }

MAKE_UNARY(log)

#undef MAKE_UNARY

}  // namespace op
}  // namespace magmadnn