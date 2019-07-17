/**
 * @file residuallayer.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-11
 *
 * @copyright Copyright (c) 2019
 */
#include <vector>
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include "layer/activation/activationlayer.h"
#include "layer/conv2d/conv2dlayer.h"
#include "layer/input/inputlayer.h"
#include "layer/layer.h"
#include "layer/shortcut/shortcutlayer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

template <typename T>
class ResidualLayer : public Layer<T> {
   public:
    ResidualLayer(op::Operation<T>* input, const std::vector<std::pair<unsigned int, unsigned int>> filters,
                  const std::vector<int> out_channels, unsigned int downsampling_stride);

    virtual ~ResidualLayer();

    virtual std::vector<op::Operation<T>*> get_weights();

   protected:
    void init();

    std::vector<std::pair<unsigned int, unsigned int>> filters;
    std::vector<int> out_channels;
    std::vector<layer::Layer<T>*> layers;
    std::vector<op::Operation<T>*> weights;

    unsigned int downsampling_stride;
};

template <typename T>
ResidualLayer<T>* residual(op::Operation<T>* input, const std::vector<std::pair<unsigned int, unsigned int>> filters,
                           const std::vector<int> out_channels, unsigned int downsampling_stride);

}  // namespace layer
}  // namespace magmadnn