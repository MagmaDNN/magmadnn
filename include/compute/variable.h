/**
 * @file variable.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-18
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <tensor/tensor.h>
#include <string>
#include "operation.h"

namespace magmadnn {
namespace op {

/** Variable Operation. The most basic operation; it simply wraps around a tensor.
 * @tparam T
 */
template <typename T>
class Variable : public Operation<T> {
   public:
    Variable(std::string name, std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type);
    Variable(std::string name, Tensor<T> *val);
    ~Variable();

    std::string to_string() { return name; }
    std::string get_name() { return name; }

   protected:
    Tensor<T> *_eval(bool recompute = true);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    std::string name;
    Tensor<T> *val;
    bool delete_tensor;
};

/** Returns a new variable operation. The variable wraps around val with name name.
 * @tparam T
 * @param name the name of the variable. This does not effect computation. It will allow to_string() methods to work,
 * however.
 * @param val tensor to wrap around
 * @return Variable<T>*
 */
template <typename T>
Variable<T> *var(std::string name, Tensor<T> *val);

/** Returns a new variable operation. This version constructs a new tensor to store in the variable.
 * @tparam T
 * @param name
 * @param shape
 * @param filler
 * @param mem_type
 * @return Variable<T>*
 */
template <typename T>
Variable<T> *var(std::string name, std::vector<unsigned int> shape, tensor_filler_t<T> filler, memory_t mem_type);

/** Creates a scalar variable.
 * @tparam T
 * @param name
 * @param val
 * @param mem_type
 * @return Variable<T>*
 */
template <typename T>
Variable<T> *scalar(std::string name, T val, memory_t mem_type);

}  // namespace op
}  // namespace magmadnn
