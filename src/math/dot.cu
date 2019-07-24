/**
 * @file dot.cu
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-07
 *
 * @copyright Copyright (c) 2019
 */
#include "math/dot.h"

namespace magmadnn {
namespace math {

#define comp_gpu(type) template void dot<GPU, type>(type, bool, const Tensor &, bool, const Tensor &, type, Tensor &);
CALL_FOR_ALL_TYPES(comp_gpu)
#undef comp_gpu

}  // namespace math
}  // namespace magmadnn