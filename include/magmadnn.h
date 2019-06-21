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


#include "types.h"
#include "init_finalize.h"
#include "utilities_internal.h"

#include "memory/memorymanager.h"
#include "tensor/tensor.h"
#include "tensor/tensor_io.h"

#include "math/tensor_math.h"

#include "compute/variable.h"
#include "compute/tensor_operations.h"
#include "compute/gradients.h"

#include "layer/layers.h"

#include "optimizer/optimizers.h"
#include "model/models.h"
