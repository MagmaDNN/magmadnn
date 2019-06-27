/**
 * @file dataloader.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-26
 * 
 * @copyright Copyright (c) 2019
 */

#pragma once
#include "tensor/tensor.h"

namespace magmadnn {
namespace dataloader {

template <typename T>
class DataLoader {
public:
    DataLoader(Tensor<T> *x, Tensor<T> *y, unsigned int batch_size): x(x), y(y), batch_size(batch_size) {
        num_batches = unsigned(x->get_shape(0) / batch_size);
        assert(num_batches > 0);
        assert(num_batches == unsigned(y->get_shape(0) / batch_size));
        
        feature_size = 1;
        for (unsigned int i = 1; i < x->get_shape().size(); i ++) {
            feature_size *= x->get_shape(i);
        }
    }
    virtual void next(Tensor<T> *x_batch, Tensor<T> *y_batch) = 0;
    virtual unsigned int get_batch_size() const {return batch_size;}
    virtual void set_batch_size(unsigned int size) {batch_size = size;}
    virtual unsigned int get_num_batches() const {return num_batches;}

protected:
    Tensor<T> *x;
    Tensor<T> *y;
    unsigned int batch_size;
    unsigned int feature_size;
    unsigned int num_batches;
};

}   // namespace dataloader
}   // namespace magmadnn