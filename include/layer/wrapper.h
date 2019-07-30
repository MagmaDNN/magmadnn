#pragma once

#include <vector>
#include "compute/operation.h"
#include "layer/layers.h"

namespace magmadnn {
namespace layer {

// template <typename T>
// using vectorOfLayer = std::vector< layer::Layer<T>* >;

template <typename T>
class layerContainer : public std::vector<Layer<T> *> {
   public:
    layerContainer(InputLayer<T> *inputlayer);
    layerContainer(op::Operation<T> *var);
    ~layerContainer();

    layer::Layer<T> *getLastLayer();
    op::Operation<T> *getLastLayerOutput();
    inline op::Operation<T> *tail() { return this->getLastLayerOutput(); };

    layerContainer<T> &appendLayer(Layer<T> *layerPtr);
    void summary(void) const;
    void destroyLayers();
};

}  // namespace layer
}  // namespace magmadnn