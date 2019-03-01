/**
 * @file input_layer.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "layer/input_layer.h"

namespace skepsi {
namespace layer {


template <typename T>
input<T>::input(std::vector<unsigned int> input_shape) : layer<T>::layer(input_shape, nullptr) {
    init();
}

template <typename T>
input<T>::input(tensor<T> *input_tensor) : layer<T>::layer(input_tensor->get_shape(), input_tensor) {
    init();
}

template <typename T>
void input<T>::init() {
    this->output_tensor = new tensor<T> (this->input_tensor->get_size(), this->input_tensor->get_memory_type());
    this->output_tensor->copy_from(this->input_tensor);
}

template <typename T>
void input<T>::forward() {
    /* does nothing */
}

template <typename T>
void input<T>::backward() {
    /* does nothing */
}

}   // layer
}   // skepsi