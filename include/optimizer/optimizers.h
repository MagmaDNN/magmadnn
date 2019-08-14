/**
 * @file optimizers.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "optimizer/adagrad/adagrad.h"
#include "optimizer/adam/adam.h"
#include "optimizer/gradientdescent/gradientdescent.h"
#include "optimizer/optimizer.h"
#include "optimizer/rmsprop/rmsprop.h"

namespace magmadnn {
namespace optimizer {

enum optimizer_t {
    SGD,
    ADAGRAD,
    RMSPROP,
    ADAM,
};

enum loss_t { CROSS_ENTROPY, MSE };

}  // namespace optimizer
}  // namespace magmadnn