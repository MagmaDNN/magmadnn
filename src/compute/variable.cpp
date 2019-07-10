/**
 * @file variable.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-18
 *
 * @copyright Copyright (c) 2019
 */
#include "compute/variable.h"

namespace magmadnn {
namespace op {

template <typename T>
Variable<T>::Variable(std::string name, std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type)
    : Operation<T>::Operation(), name(name) {
    val = new Tensor<T>(shape, filler, mem_type);
    delete_tensor = true;

    this->output_shape = val->get_shape();
    this->mem_type = val->get_memory_type();

    /* so eval doesn't return NULL */
    this->output_tensor = val;
    this->has_been_computed = true;
}
template <typename T>
Variable<T>::Variable(std::string name, Tensor<T>* val) : Operation<T>::Operation(), name(name), val(val) {
    this->output_shape = val->get_shape();
    this->mem_type = val->get_memory_type();
    delete_tensor = false;

    /* so eval doesn't return NULL */
    this->output_tensor = val;
    this->has_been_computed = true;
}

template <typename T>
Variable<T>::~Variable() {
    if (delete_tensor) delete val;
}

template <typename T>
Tensor<T>* Variable<T>::_eval(bool recompute) {
    return val;
}

template <typename T>
Tensor<T>* Variable<T>::_grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad) {
    /* TODO : if (var == this) return 1; */

    Tensor<T>* out;

    if (var == this) {
        out = this->_grad_cache[(uintptr_t) var];
        if (out == NULL) {
            out = new Tensor<T>({1}, {ONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) var] = out;
        }
        return out;
    }

    return grad;
}
// compile for int, float, double
template class Variable<int>;
template class Variable<float>;
template class Variable<double>;

template <typename T>
Variable<T>* var(std::string name, Tensor<T>* val) {
    return new Variable<T>(name, val);
}
template Variable<int>* var(std::string name, Tensor<int>* val);
template Variable<float>* var(std::string name, Tensor<float>* val);
template Variable<double>* var(std::string name, Tensor<double>* val);

template <typename T>
Variable<T>* var(std::string name, std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type) {
    return new Variable<T>(name, shape, filler, mem_type);
}
template Variable<int>* var(std::string, std::vector<unsigned int>, tensor_filler_t<int>, memory_t mem_type);
template Variable<float>* var(std::string, std::vector<unsigned int>, tensor_filler_t<float>, memory_t mem_type);
template Variable<double>* var(std::string, std::vector<unsigned int>, tensor_filler_t<double>, memory_t mem_type);

template <typename T>
Variable<T>* scalar(std::string name, T val, memory_t mem_type) {
    return new Variable<T>(name, {1}, {CONSTANT, {val}}, mem_type);
}
template Variable<int>* scalar(std::string, int, memory_t mem_type);
template Variable<float>* scalar(std::string, float, memory_t mem_type);
template Variable<double>* scalar(std::string, double, memory_t mem_type);

}  // namespace op
}  // namespace magmadnn