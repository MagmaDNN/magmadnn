/**
 * @file sumop.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-20
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cstdio>
#include <vector>
#include "compute/compute_graph.h"
#include "compute/operation.h"
#include "compute/sum/sum_internal.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

namespace magmadnn {
namespace op {

class SumOp : public Operation {
   public:
    SumOp(std::vector<Operation *> ops);

    std::string to_string() const override;

   protected:
    Tensor &_eval(bool recompute = true) override;
    Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) override;

    std::vector<Operation *> ops;
};

inline Operation *sum(std::vector<Operation *> ops) { return default_graph.add_operation<SumOp>(ops); }

}  // namespace op
}  // namespace magmadnn