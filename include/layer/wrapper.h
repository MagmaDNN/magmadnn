#pragma once

#include <vector>
#include "layer/layers.h"
#include "compute/operation.h"

namespace magmadnn {
namespace layer {

// template <typename T>
// using vectorOfLayer = std::vector< layer::Layer<T>* >;

template <typename T>
class layerContainer : public std::vector<Layer<T>* > {
public:
    layerContainer(InputLayer<T> *inputlayer);
    layerContainer(op::Operation<T> *var);
    ~layerContainer();

    layer::Layer<T> *getLastLayer();
    op::Operation<T> *getLastLayerOutput();
    inline op::Operation<T> *tail() { return this->getLastLayerOutput(); };

    layerContainer<T> &appendLayer(Layer<T> *layerPtr);

    void destroyLayers();
};

}   // layer
}   // magmadnn