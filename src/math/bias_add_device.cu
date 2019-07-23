/**
 * @file bias_add_device.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-23
 *
 * @copyright Copyright (c) 2019
 */
#include "math/bias_add.h"

#define BLK_SIZE 1024

namespace magmadnn {
namespace math {

template <typename T>
__global__ void kernel_bias_add_device(const T *x, const T *bias, T *out, unsigned int x_rows, unsigned int x_cols) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < x_cols * x_rows; i += stride) {
        out[i] = x[i] + bias[i / x_cols];
    }
}

template <typename T>
void bias_add_device(const Tensor &x, const Tensor &bias, Tensor &out) {
    unsigned int x_rows = x.shape(0);
    unsigned int x_cols = x.shape(1);

    kernel_bias_add_device<<<(x_rows * x_cols + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE>>>(x.get_ptr<T>(), bias.get_ptr<T>(),
                                                                                      out.get_ptr<T>(), x_rows, x_cols);
}
#define COMPILE_BIASADD_DEVICE(type) template void bias_add_device<type>(const Tensor&, const Tensor&, Tensor&);
CALL_FOR_ALL_TYPES(COMPILE_BIASADD_DEVICE)
#undef COMPILE_BIASADD_DEVICE

}  // namespace math
}  // namespace magmadnn

#undef BLK_SIZE
