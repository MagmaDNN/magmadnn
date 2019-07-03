/**
 * @file init_finalize.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-03-15
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "types.h"

#if defined(_HAS_CUDA_)
#include "magma.h"
#include "cublas_v2.h"
#endif

namespace magmadnn {

/** Should be called at the start of every program.
 * @return magmadnn_error_t 
 */
magmadnn_error_t magmadnn_init();

/** Cleanup. Should be called at the end of every program.
 * @return magmadnn_error_t 
 */
magmadnn_error_t magmadnn_finalize();

}   // magmadnn