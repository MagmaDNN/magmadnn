/**
 * @file tensor.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 * 
 * @copyright Copyright (c) 2019
 */
#include "tensor.h"


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
void tensor<T>::set(const std::vector<int>& idx, T val) { }


template <typename T>
void tensor<T>::init(std::vector<int>& shape, T fill, device_t device_id) { }

template <typename T>
int tensor<T>::get_flattened_index(const std::vector<int>& idx) {
    // TODO
    return 0;
 }



/* COMPILE FOR INT, FLOAT, AND DOUBLE */
template class tensor<int>;
template class tensor<float>;
template class tensor<double>;