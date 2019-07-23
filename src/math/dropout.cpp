/**
 * @file dropout.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-07-01
 *
 * @copyright Copyright (c) 2019
 */
#include "math/dropout.h"

namespace magmadnn {
namespace math {

template <typename T>
void dropout(const Tensor &x, Tensor &out, const Tensor &mask, float dropout_rate) {
    if (out.get_memory_type() == HOST) {
        // float p = 1.0f - dropout_rate;
        /* TODO -- generate mask in dropout op */
        // Tensor<T> a(mask->get_shape(), {MASK, {static_cast<double>(p), static_cast<T>(1.0f / p)}},
        // mask->get_memory_type()); mask->copy_from(a);
        // math::product<T>(mask, x, out);

        /* compute the product between x and mask into out */
        math::launchMappedKernelCPU<product_map, T>(out.size(), x.get_ptr<T>(), mask.get_ptr<T>(), out.get_ptr<T>());
    } else {
        LOG(ERROR) << "For dropout on GPU, please use dropout_device\n";
    }
}
#define COMPILE_DROPOUT(type) template void dropout<type>(const Tensor &, Tensor &, const Tensor &, float);
CALL_FOR_ALL_TYPES(COMPILE_DROPOUT)
#undef COMPILE_DROPOUT

template <typename T>
void dropout_grad(const Tensor &grad, Tensor &out, const Tensor &mask) {
    if (out.get_memory_type() == HOST) {
        // math::product<T>(mask, grad, out);

        /* compute the product between grad and mask into out */
        math::launchMappedKernelCPU<product_map, T>(out.size(), grad.get_ptr<T>(), mask.get_ptr<T>(), out.get_ptr<T>());
    }
#if defined(_HAS_CUDA_)
    else {
        fprintf(stderr, "For dropout_grad on GPU, please use dropout_grad_device\n");
    }
#endif
}
#define COMPILE_DROPOUTGRAD(type) template void dropout_grad<type>(const Tensor &, Tensor &, const Tensor &);
CALL_FOR_ALL_TYPES(COMPILE_DROPOUTGRAD)
#undef COMPILE_DROPOUTGRAD

#if defined(_HAS_CUDA_)
template <typename T>
void dropout_device(const Tensor &x, Tensor &out, cudnn_dropout_settings_t settings,
                    cudnn_dropout_shared_settings_t shared) {
    cudnnErrchk(cudnnDropoutForward(settings.handle, shared.dropoutDesc, settings.xdesc, (void *) x.get_ptr<T>(),
                                    settings.ydesc, (void *) out.get_ptr<T>(), shared.reserveSpace,
                                    shared.reserveSpaceSizeInBytes));
}
#define COMPILE_DROPOUT_DEVICE(type)                                                       \
    template void dropout_device<type>(const Tensor &, Tensor &, cudnn_dropout_settings_t, \
                                       cudnn_dropout_shared_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_DROPOUT_DEVICE)
#undef COMPILE_DROPOUT_DEVICE

template <typename T>
void dropout_grad_device(const Tensor &grad, Tensor &out, cudnn_dropout_grad_settings_t settings,
                         cudnn_dropout_shared_settings_t shared) {
    cudnnErrchk(cudnnDropoutBackward(settings.handle, shared.dropoutDesc, settings.dydesc, (void *) grad.get_ptr<T>(),
                                     settings.dxdesc, (void *) out.get_ptr<T>(), shared.reserveSpace,
                                     shared.reserveSpaceSizeInBytes));
}
#define COMPILE_DROPOUTGRAD_DEVICE(type)                                                             \
    template void dropout_grad_device<type>(const Tensor &, Tensor &, cudnn_dropout_grad_settings_t, \
                                            cudnn_dropout_shared_settings_t);
CALL_FOR_ALL_TYPES(COMPILE_DROPOUTGRAD_DEVICE)
#undef COMPILE_DROPOUTGRAD_DEVICE

#endif

}  // namespace math
}  // namespace magmadnn