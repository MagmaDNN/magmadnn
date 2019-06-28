/**
 * @file linearloader.h
 * @author Sedrick Keh
 * @version 0.1
 * @date 2019-06-26
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include "dataloader/dataloader.h"

namespace magmadnn {
namespace dataloader {

template <typename T>
class LinearLoader : public DataLoader<T> {
public:
    LinearLoader(Tensor<T> *x, Tensor<T> *y, unsigned int batch_size);
    
    virtual void next(Tensor<T> *x_batch, Tensor<T> *y_batch);

    virtual void reset();
    
private:
    unsigned int curr_index;
};

}   // namespace dataloader
}   // namespace magmadnn

