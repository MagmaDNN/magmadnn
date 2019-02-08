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
tensor<T>::tensor(std::vector<int> shape) { }

template <typename T>
tensor<T>::tensor(std::vector<int> shape, device_t device_id) { }

template <typename T>
tensor<T>::tensor(std::vector<int> shape, T fill) { }


template <typename T>
tensor<T>::tensor(std::vector<int> shape, T fill, device_t device_id) { }

template <typename T>
tensor<T>::~tensor() { }


template <typename T>
T tensor<T>::get(const std::vector<int>& idx) { 
    // TODO
    return (T) 0;
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
    // TODO
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