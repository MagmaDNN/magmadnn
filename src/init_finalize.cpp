/**
 * @file init_finalize.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-03-15
 * 
 * @copyright Copyright (c) 2019
 */
#include "init_finalize.h"

namespace magmadnn {

magmadnn_error_t magmadnn_init() {
    magmadnn_error_t err = 0;

    #if defined(_HAS_CUDA_)
    err = (magmadnn_error_t) magma_init();
    #endif

    return err;
}

magmadnn_error_t magmadnn_finalize() {
    magmadnn_error_t err = 0;

    #if defined(_HAS_CUDA_)
    err = (magmadnn_error_t) magma_finalize();
    #endif

    return err;
}

}   // namespace magmadnn