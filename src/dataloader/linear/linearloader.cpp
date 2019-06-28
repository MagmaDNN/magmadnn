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
    unsigned int batch_mem_space_x = this->batch_size * this->sample_size_x;
    unsigned int batch_mem_space_y = this->batch_size * this->sample_size_y;
    assert((curr_index + 1) * batch_mem_space_x <= this->x->get_size());
    assert((curr_index + 1) * batch_mem_space_y <= this->y->get_size());
    x_batch->copy_from(*(this->x), curr_index * batch_mem_space_x, batch_mem_space_x);
    y_batch->copy_from(*(this->y), curr_index * batch_mem_space_y, batch_mem_space_y);
    curr_index ++;
}

template class LinearLoader<int>;
template class LinearLoader<float>;
template class LinearLoader<double>;
}   // namespace dataloader
}   // namespace magmadnn