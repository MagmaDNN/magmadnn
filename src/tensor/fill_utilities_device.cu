/**
 * @file fill_utilities_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-12
 * 
 * @copyright Copyright (c) 2019
 */

#include "tensor/fill_utilities.h"

#ifdef _HAS_CUDA_

namespace skepsi {

template <typename T>
__global__ void kernel_fill_glorot(T *arr, double *vals) {

}

} // namespace skepsi
#endif
