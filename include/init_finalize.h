/**
 * @file init_finalize.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-03-15
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "types.h"

#if defined(_HAS_CUDA_)
#include "cublas_v2.h"
#include "magma.h"
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

}  // namespace magmadnn