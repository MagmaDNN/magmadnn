/**
 * @file dot.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-06
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "data_types.h"
#include "math/binary_math_operations.h"
#include "math/launch_math_kernel.h"
#include "math/matmul.h"
#include "mdnn_device_types.h"
#include "tensor/tensor.h"
#include "tensor/tensor_utilities.h"

namespace magmadnn {
namespace math {

template <DeviceType dev, typename T>
void dot(T alpha, bool trans_A, const Tensor &A, bool trans_B, const Tensor &B, T beta, Tensor &out) {
    MAGMADNN_ASSERT(
        ::magmadnn::utilities::do_tensors_match(GetDataType<T>::value, GetMemoryType<dev>::value, {A, B, out}),
        "tensors must match in data type and memory type");

    size_t n_axes_a = A.shape().size();
    size_t n_axes_b = B.shape().size();
    bool a_is_scalar = (A.size() == 1);
    bool b_is_scalar = (B.size() == 1);

    if (n_axes_a == 2 && n_axes_b == 2) {
        /* matmul(A,B) */

        matmul<T>(alpha, trans_A, A, trans_B, B, beta, out);

    } else if (n_axes_a == 2 && n_axes_b == 1 && !b_is_scalar) {
        /* gemv(A,B) */
        LOG(ERROR) << "add gemv\n";
    } else if (n_axes_a == 1 && n_axes_b == 2 && !a_is_scalar) {
        /* gemv(B^T,A) */
        LOG(ERROR) << "add gemv\n";
    } else if (a_is_scalar && !b_is_scalar) {
        /* broadcast product - a(scalar) b(tensor) */

        A.get_memory_manager()->sync();

        ParallelLauncher<dev, scalar_tensor_product_map>::launchMappedKernel(out.size(), A.get<T>(0), B.get_ptr<T>(),
                                                                             out.get_ptr<T>());

    } else if (b_is_scalar) {
        /* broadcast product - a(scalar?OR?tensor) b(scalar) */

        B.get_memory_manager()->sync();

        ParallelLauncher<dev, scalar_tensor_product_map>::launchMappedKernel(out.size(), B.get<T>(0), A.get_ptr<T>(),
                                                                             out.get_ptr<T>());

    } else {
        /* other */
        LOG(ERROR) << "undefined dot product";
    }
}

template <DeviceType dev, typename T>
void dot(T alpha, const Tensor &A, const Tensor &B, T beta, Tensor &out) {
    dot<dev>(alpha, false, A, false, B, beta, out);
}

template <DeviceType dev>
void dot(const Tensor &A, const Tensor &B, Tensor &out) {
    FOR_ALL_DTYPES(out.dtype(), Dtype, {
        // compute the dot product for each data type
        dot<CPU>(static_cast<Dtype>(1), A, B, static_cast<Dtype>(0), out);
    })
}

}  // namespace math
}  // namespace magmadnn