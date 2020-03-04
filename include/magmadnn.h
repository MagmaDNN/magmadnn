/**
 * @file magmadnn.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

/* include all magmadnn header files */

// TODO remove the following define and rely on the config.h file
// instead
#if defined(USE_GPU)
#define _HAS_CUDA_
#endif

// TODO remove the following define
#if defined(_HAS_CUDA_)
#define USE_GPU
#endif

#include "magmadnn/config.h"
#include "magmadnn/init_finalize.h"
#include "magmadnn/types.h"
#include "magmadnn/utilities_internal.h"

#include "magmadnn/data/Dataset.h"
#include "magmadnn/data/MNIST.h"
#include "magmadnn/data/CIFAR10.h"

#include "magmadnn/exception.h"
#include "magmadnn/exception_helpers.h"

#include "memory/memorymanager.h"
#include "tensor/tensor.h"
#include "tensor/tensor_io.h"

#include "math/tensor_math.h"

#include "compute/gradients.h"
#include "compute/tensor_operations.h"
#include "compute/variable.h"

#include "layer/layers.h"

#include "model/models.h"
#include "optimizer/optimizers.h"

#include "dataloader/dataloaders.h"
