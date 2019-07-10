/**
 * @file optimizers.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "optimizer/gradientdescent/gradientdescent.h"
#include "optimizer/optimizer.h"

namespace magmadnn {
namespace optimizer {

enum optimizer_t {
    SGD,
    ADAM,
};

enum loss_t { CROSS_ENTROPY, MSE };

}  // namespace optimizer
}  // namespace magmadnn