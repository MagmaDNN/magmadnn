/**
 * @file binary_math_operations.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-07-22
 *
 * @copyright Copyright (c) 2019
 */

#include "types.h"

namespace magmadnn {
namespace math {

#define NEW_BINARY_KERNEL(name, expr)                                         \
    struct name##_map {                                                       \
        template <typename Dtype>                                             \
        inline static void map(index_t idx, Dtype *a, Dtype *b, Dtype *out) { \
            expr;                                                             \
        }                                                                     \
    }

NEW_BINARY_KERNEL(add, out[idx] = a[idx] + b[idx]);
NEW_BINARY_KERNEL(sub, out[idx] = a[idx] - b[idx]);
NEW_BINARY_KERNEL(product, out[idx] = a[idx] * b[idx]);
NEW_BINARY_KERNEL(div, out[idx] = a[idx] / b[idx]);

}  // namespace math
}  // namespace magmadnn