/**
 * @file binaryop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-08-08
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "compute/compute_graph.h"
#include "compute/operation.h"

#include "magmadnn_device_types.h"

#include "math/binary_math_operations.h"
#include "math/launch_math_kernel.h"

namespace magmadnn {
namespace op {

template <typename BinaryOpType>
class BinaryOp : public Operation {
   public:
    BinaryOp(Operation *x, Operation *y) {
        this->use_tensor_settings(x, true);

        this->output_tensor_ = Tensor(this->output_shape_, this->dtype_, {NONE}, this->mem_type_);
    }

    std::string to_string() const override { return "BIN_OP(" + x->to_string() + ", " + y->to_string() + ")"; }

   protected:
    Tensor &_eval(bool recompute = true) override {
        Tensor &x_tensor = x->eval(recompute);
        Tensor &y_tensor = y->eval(recompute);

        FOR_ALL_DEVICE_TYPES(getDeviceType(this->mem_type_), DEV_TYPE, {
            ::magmadnn::math::ParallelLauncher<DEV_TYPE, BinaryOpType>(x_tensor, y_tensor, this->output_tensor_);
        })
    }

    Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) override {}

    Operation *x, *y;
};

#define MAKE_BINARY(name)                                                                 \
    inline Operation *name(Operation *a, Operation *b) {                                  \
        return default_graph.add_operation<BinaryOp<::magmadnn::math::name##_map>>(a, b); \
    }

MAKE_BINARY(add)
MAKE_BINARY(sub)
MAKE_BINARY(product)
MAKE_BINARY(div)
MAKE_BINARY(pow)

#undef MAKE_BINARY

}  // namespace op
}  // namespace magmadnn