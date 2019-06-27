/**
 * @file linearloader.cpp
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-26
 * 
 * @copyright Copyright (c) 2019
 */
#include "dataloader/linear/linearloader.h"

namespace magmadnn {
namespace dataloader {

template <typename T>
LinearLoader<T>::LinearLoader(Tensor<T> *x, Tensor<T> *y, unsigned int batch_size) :
    DataLoader<T>::DataLoader(x, y, batch_size), curr_index(0) {}

template <typename T>
void LinearLoader<T>::next(Tensor<T> *x_batch, Tensor<T> *y_batch) {
    unsigned int batch_mem_space = this->batch_size * this->feature_size;
    assert((curr_index + 1) * batch_mem_space <= this->x->get_size());
    assert((curr_index + 1) * this->batch_size <= this->y->get_size());
    x_batch->copy_from(*(this->x), curr_index * batch_mem_space, batch_mem_space);
    y_batch->copy_from(*(this->y), curr_index * this->batch_size, this->batch_size);
    curr_index ++;
}

template class LinearLoader<int>;
template class LinearLoader<float>;
template class LinearLoader<double>;
}   // namespace dataloader
}   // namespace magmadnn