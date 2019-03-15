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
#endif

namespace skepsi {

/** Should be called at the start of every program.
 * @return skepsi_error_t 
 */
skepsi_error_t skepsi_init();

/** Cleanup. Should be called at the end of every program.
 * @return skepsi_error_t 
 */
skepsi_error_t skepsi_finalize();

}   // skepsi