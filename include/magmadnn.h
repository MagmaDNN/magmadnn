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

#if defined(USE_GPU)
#define _HAS_CUDA_
#endif

#if defined(_HAS_CUDA_)
#define USE_GPU
#endif

#include "init_finalize.h"
#include "types.h"
#include "utilities_internal.h"

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