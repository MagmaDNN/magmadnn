/**
 * @file shortcutlayer.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-26
 *
 * @copyright Copyright (c) 2019
 */
#include "layer/shortcut/shortcutlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
ShortcutLayer<T>::ShortcutLayer(op::Operation<T> *input, op::Operation<T> *skip)
    : Layer<T>::Layer(input->get_output_shape(), input), skip(skip) {
    init();
}

template <typename T>
std::vector<op::Operation<T> *> ShortcutLayer<T>::get_weights() {
    return {};
}

template <typename T>
ShortcutLayer<T>::~ShortcutLayer() {}

template <typename T>
void ShortcutLayer<T>::init() {
    this->name = "ShortcutLayer";

    this->output = op::add(this->input, this->skip);
}
template class ShortcutLayer<int>;
template class ShortcutLayer<float>;
template class ShortcutLayer<double>;

template <typename T>
ShortcutLayer<T> *shortcut(op::Operation<T> *input, op::Operation<T> *skip) {
    return new ShortcutLayer<T>(input, skip);
}
template ShortcutLayer<int> *shortcut(op::Operation<int> *, op::Operation<int> *);
template ShortcutLayer<float> *shortcut(op::Operation<float> *, op::Operation<float> *);
template ShortcutLayer<double> *shortcut(op::Operation<double> *, op::Operation<double> *);

}  // namespace layer
}  // namespace magmadnn