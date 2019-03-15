/**
 * @file init_finalize.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-03-15
 * 
 * @copyright Copyright (c) 2019
 */
#include "init_finalize.h"

namespace skepsi {

skepsi_error_t skepsi_init() {
    skepsi_error_t err = 0;

    #if defined(_HAS_CUDA_)
    err = (skepsi_error_t) magma_init();
    #endif

    return err;
}

skepsi_error_t skepsi_finalize() {
    skepsi_error_t err = 0;

    #if defined(_HAS_CUDA_)
    err = (skepsi_error_t) magma_finalize();
    #endif

    return err;
}

}   // namespace skepsi