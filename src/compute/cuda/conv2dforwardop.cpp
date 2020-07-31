#include "compute/conv2dforward/conv2dforwardop.h"

#include <iostream>

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif

namespace magmadnn {
namespace op {

template <typename T>
void Conv2DForwardOp<T>::cuda_forward() {
    this->cudnn_settings.handle = this->get_cudnn_handle();
    ::magmadnn::math::conv2d_device(this->input_tensor, this->filter_tensor, this->output_tensor, this->cudnn_settings);
    if (!this->get_async()) {
        cudaStreamSynchronize(this->get_custream());
    }
}

template class Conv2DForwardOp<int>;
template class Conv2DForwardOp<float>;
template class Conv2DForwardOp<double>;

}  // namespace op
}  // namespace magmadnn
