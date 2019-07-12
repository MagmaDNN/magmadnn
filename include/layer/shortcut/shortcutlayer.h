/**
 * @file shortcutlayer.h
 * @author Sedrick Keh
 * @version 1.0
 * @date 2019-07-11
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <vector>
#include "compute/add/addop.h"
#include "compute/operation.h"
#include "layer/layer.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace layer {

template <typename T>
class ShortcutLayer : public Layer<T> {
   public:
    ShortcutLayer(op::Operation<T>* input, op::Operation<T>* skip);
    virtual ~ShortcutLayer();

    virtual std::vector<op::Operation<T>*> get_weights();

   protected:
    void init();
    op::Operation<T>* skip;
};

template <typename T>
ShortcutLayer<T>* shortcut(op::Operation<T>* input, op::Operation<T>* skip);

}  // namespace layer
}  // namespace magmadnn