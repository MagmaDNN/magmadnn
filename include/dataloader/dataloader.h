/**
 * @file dataloader.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-26
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <cassert>

#include "tensor/tensor.h"

namespace magmadnn {
namespace dataloader {

template <typename T>
class DataLoader {
   public:
    /** Constructs a DataLoader object with given data and parameters
     * @param x input features
     * @param y output labels
     * @param batch_size
     */
    DataLoader(Tensor<T> *x, Tensor<T> *y, unsigned int batch_size) : x(x), y(y), batch_size(batch_size) {
        num_batches = unsigned(x->get_shape(0) / batch_size);
        assert(num_batches > 0);
        assert(num_batches == unsigned(y->get_shape(0) / batch_size));

        sample_size_x = x->get_size() / x->get_shape(0);
        sample_size_y = y->get_size() / y->get_shape(0);
    }

    /** Copies the next batch of inputs and outputs to x_batch and y_batch
     * @param x_batch
     * @param y_batch
     */
    virtual void next(Tensor<T> *x_batch, Tensor<T> *y_batch) = 0;

    /** Resets the dataloader for the next epoch
     */
    virtual void reset() = 0;

    virtual unsigned int get_batch_size() const { return batch_size; }
    virtual void set_batch_size(unsigned int size) {
        batch_size = size;
        num_batches = unsigned(x->get_shape(0) / batch_size);
        assert(num_batches > 0);
        assert(num_batches == unsigned(y->get_shape(0) / batch_size));
    }
    virtual unsigned int get_num_batches() const { return num_batches; }

   protected:
    Tensor<T> *x;
    Tensor<T> *y;
    unsigned int batch_size;
    unsigned int sample_size_x;
    unsigned int sample_size_y;
    unsigned int num_batches;
};

}  // namespace dataloader
}  // namespace magmadnn
