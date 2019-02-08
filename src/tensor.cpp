/**
 * @file tensor.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "tensor.h"

namespace skepsi {

template <typename T>
tensor<T>::tensor(std::vector<unsigned int> shape) {
    init(shape, TENSOR_DEFAULT_FILLER, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
 }

 template <typename T>
tensor<T>::tensor(std::vector<unsigned int> shape, memory_t mem_type) {
    init(shape, TENSOR_DEFAULT_FILLER, mem_type, TENSOR_DEFAULT_DEVICE_ID);
 }

template <typename T>
tensor<T>::tensor(std::vector<unsigned int> shape, memory_t mem_type, device_t device_id) { 
    init(shape, TENSOR_DEFAULT_FILLER, mem_type, device_id);
}

template <typename T>
tensor<T>::tensor(std::vector<unsigned int> shape, tensor_filler_t filler) { 
    init(shape, filler, TENSOR_DEFAULT_MEM_TYPE, TENSOR_DEFAULT_DEVICE_ID);
}

template <typename T>
tensor<T>::tensor(std::vector<unsigned int> shape, tensor_filler_t filler, memory_t mem_type) { 
    init(shape, filler, mem_type, TENSOR_DEFAULT_DEVICE_ID);
}


template <typename T>
tensor<T>::tensor(std::vector<unsigned int> shape, tensor_filler_t filler, memory_t mem_type, device_t device_id) { 
    init(shape, filler, mem_type, device_id);
}

template <typename T>
tensor<T>::~tensor() { 
    delete mem_manager;
}


template <typename T>
void tensor<T>::init(std::vector<unsigned int>& shape, tensor_filler_t filler, memory_t mem_type, device_t device_id) {
    // tensor must have at least 1 axis
    assert( shape.size() != 0 );

    // initialize class variables
    this->shape = shape;
    this->mem_type = mem_type;
    this->device_id = device_id;

    // calculate the total number of elements
    this->size = 1;
    for (unsigned int i = 0; i < shape.size(); i++) {
        this->size *= shape[i];
    }

    // create memory manager
    this->mem_manager = new memorymanager<T> (size, mem_type, device_id);

}


template <typename T>
T tensor<T>::get(const std::vector<int>& idx) { 
    return mem_manager->get( get_flattened_index(idx) );
}

template <typename T>
void tensor<T>::set(const std::vector<int>& idx, T val) { 
    mem_manager->set( get_flattened_index(idx), val );
}

template <typename T>
unsigned int tensor<T>::get_flattened_index(const std::vector<int>& idx) {
    unsigned int jump_size = 1; // the total amout to jump to get to next axis
    unsigned int flattened_idx = 0;

    for (int i = ((int)idx.size()) - 1; i >= 0; i--) {
        flattened_idx += idx[i] * jump_size;
        jump_size *= shape[i];
    }
    return flattened_idx;
 }



/* COMPILE FOR INT, FLOAT, AND DOUBLE */
template class tensor<int>;
template class tensor<float>;
template class tensor<double>;

} // namespace skepsi