/**
 * @file argmax.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-06-11
 * 
 * @copyright Copyright (c) 2019
 */
#include "math/argmax.h"

namespace magmadnn {
namespace math {

template <typename T>
void argmax(Tensor<T> *x, int axis, Tensor<T> *out) {
    unsigned int x_n_axes = x->get_shape().size();
    
    assert( out->get_shape().size() == (x_n_axes-1) );

    T max, val, arg_max;
    if (x_n_axes > 2) {
        /* TODO -- implement this */
        fprintf(stderr, "argmax for tensors with more than 2 axes not supported.\n");
        return;
    } else if (x_n_axes == 2) {
        for (unsigned int i = 0; i < x->get_shape(0); i++) {
            max = x->get({i,(unsigned int)0});
            arg_max = (T)0;
            for (unsigned int j = 1; j < x->get_shape(1); j++) {
                val = x->get({i, j});
                if (val > max) {
                    arg_max = (T) j;
                    max = val;
                }
            }
            out->set({i}, arg_max);
        }
    } else {
        max = x->get({0});
        arg_max = (T)0;
        for (unsigned int i = 1; i < x->get_shape(0); i++) {
            val = x->get({i});
            if (val > max) {
                arg_max = (T) i;
                max = val;
            }
        }
        out->set({0}, arg_max);
    }
}
template void argmax(Tensor<int> *x, int axis, Tensor<int> *out);
template void argmax(Tensor<float> *x, int axis, Tensor<float> *out);
template void argmax(Tensor<double> *x, int axis, Tensor<double> *out);

}
}