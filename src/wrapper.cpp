#include "wrapper.h"

namespace magmadnn{

template <typename T>
layerContainer<T>::layerContainer(layer::InputLayer<T> *inputlayer){
    this->push_back(inputlayer);
}
template <typename T>
layerContainer<T>::layerContainer(op::Operation<T> *var){
    this->push_back(layer::input(var));
}

template <typename T>
layerContainer<T>::~layerContainer(){
    this->destroyLayers();
}
template <typename T>
layer::Layer<T> *layerContainer<T>::getLastLayer(void){
    return this->back();
}
template <typename T>
op::Operation<T> *layerContainer<T>::getLastLayerOutput(void) {
    return this->back()->out();
}

template <typename T>
layerContainer<T> &layerContainer<T>::appendLayer(layer::Layer<T> *layerPtr){
    this->push_back(layerPtr);
    return *this;
}
template <typename T>
void layerContainer<T>::destroyLayers(void){
    for (auto ptr = this->rbegin(); ptr != this->rend(); ++ptr){
        delete *ptr;
        *ptr = nullptr;
    }
    this->clear();
}

template class layerContainer <int>;
template class layerContainer <float>;
template class layerContainer <double>;

}   // magmadnn

