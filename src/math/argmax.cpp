/**
 * @file argmax.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-11
 *
 * @copyright Copyright (c) 2019
 */
#include "math/argmax.h"

namespace magmadnn {
namespace math {

template <typename T>
void argmax(const Tensor &x, int axis, Tensor &out) {
    const std::vector<index_t> &x_shape = x.shape();
    const std::vector<index_t> &out_shape = out.shape();
    unsigned int x_n_axes = x_shape.size();

    assert(out_shape.size() == (x_n_axes - 1));
    assert(out.size() == x_shape[axis]);
    // assert(T_IS_SAME_MEMORY_TYPE(x, out));
    MAGMADNN_ASSERT(::magmadnn::utilities::do_tensors_match(GetDataType<T>::value, out.get_memory_type(), {x, out}),
                    "invalid tensors");

    if (out.get_memory_type() == HOST) {
        T max, val, arg_max;
        const T *x_ptr = x.get_ptr<T>();
        T *out_ptr = out.get_ptr<T>();

        if (x_n_axes > 2) {
            /* TODO -- implement this */
            LOG(ERROR) << "argmax for tensors with more than 2 axes not supported.\n";
            return;
        } else if (x_n_axes == 2) {
            /* ARG_MAX OF EACH ROW */

            for (unsigned int i = 0; i < x_shape[(axis == 0) ? 0 : 1]; i++) {
                if (axis == 0)
                    max = x_ptr[i * x_shape[1]]; /* first element of row i */
                else
                    max = x_ptr[i]; /* first element of column i */

                arg_max = static_cast<T>(0);
                for (unsigned int j = 1; j < x_shape[(axis == 0) ? 1 : 0]; j++) {
                    if (axis == 0)
                        val = x_ptr[i * x_shape[1] + j];  // x[i,j]
                    else
                        val = x_ptr[j * x_shape[1] + i];  // x[j,i]

                    if (val > max) {
                        arg_max = (T) j;
                        max = val;
                    }
                }
                out_ptr[i] = arg_max;
            }
        } else {
            max = x_ptr[0];
            arg_max = (T) 0;
            for (unsigned int i = 1; i < x_shape[0]; i++) {
                val = x_ptr[i];
                if (val > max) {
                    arg_max = (T) i;
                    max = val;
                }
            }
            out_ptr[0] = arg_max;
        }
    }
#if defined(_HAS_CUDA_)
    else {
        LOG(ERROR) << "Argmax on GPU not yet defined.\n";
    }
#endif
}
#define comp(type) template void argmax<type>(const Tensor &, int, Tensor &);
CALL_FOR_ALL_TYPES(comp)
#undef comp

}  // namespace math
}  // namespace magmadnn