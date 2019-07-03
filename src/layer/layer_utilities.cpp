/**
 * @file layer_utilities.h
 * @author Daniel Nichols
 * @version 0.1.0
 * @date 2019-07-03
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/layer_utilities.h"

namespace magmadnn {
namespace layer {
namespace utilities {


template <typename T>
magmadnn_error_t copy_layer(layer::Layer<T> *dst, layer::Layer<T> *src) {
    const std::vector<::magmadnn::op::Operation<T> *>& dst_weights = dst->get_weights();
    const std::vector<::magmadnn::op::Operation<T> *>& src_weights = src->get_weights();

    /* ensure we have the same number of weights to copy */
    assert(dst_weights.size() == src_weights.size());

    for (unsigned int i = 0; i < dst_weights.size(); i++) {
        dst_weights[i]->get_output_tensor()->copy_from(*src_weights[i]->get_output_tensor());
    }
    return (magmadnn_error_t) 0;
}

template <typename T>
magmadnn_error_t copy_layers(std::vector<layer::Layer<T> *> dsts, std::vector<layer::Layer<T> *> srcs) {
    assert( dsts.size() == srcs.size());

    for (unsigned int i = 0; i < dsts.size(); i++) {
        copy_layer(dsts[i], srcs[i]);
    }

    return (magmadnn_error_t) 0;
}
template magmadnn_error_t copy_layers(std::vector<layer::Layer<int> *> dsts, std::vector<layer::Layer<int> *> srcs);
template magmadnn_error_t copy_layers(std::vector<layer::Layer<float> *> dsts, std::vector<layer::Layer<float> *> srcs);
template magmadnn_error_t copy_layers(std::vector<layer::Layer<double> *> dsts, std::vector<layer::Layer<double> *> srcs);

}
}
}