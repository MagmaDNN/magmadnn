#include <vector>
#include "layer/layers.h"
#include "compute/operation.h"

namespace magmadnn{

// template <typename T>
// using vectorOfLayer = std::vector< layer::Layer<T>* >;

template <typename T>
class layerContainer : public std::vector< layer::Layer<T>* > {
    public:
    layerContainer(layer::InputLayer<T> *inputlayer);
    layerContainer(op::Operation<T> *var);
    ~layerContainer();

    layer::Layer<T> *getLastLayer(void);
    op::Operation<T> *getLastLayerOutput(void);
    inline op::Operation<T> *tail(void){return this->getLastLayerOutput();};

    layerContainer<T> &appendLayer(layer::Layer<T> *layerPtr);

    void destroyLayers(void);
};

}   // magmadnn