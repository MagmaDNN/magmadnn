/**
 * @file variable.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-18
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include <string>

#include "compute/compute_graph.h"
#include "data_types.h"
#include "operation.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

/** Variable Operation. The most basic operation; it simply wraps around a tensor.
 * @tparam T
 */
class Variable : public Operation {
   public:
    Variable(std::string name, const std::vector<index_t> &shape, DataType dtype = FLOAT,
             tensor_filler_t filler = {NONE}, memory_t mem_type = HOST);
    Variable(std::string name, Tensor &val);

    std::string to_string() const override { return name_; }
    std::string get_name() const override { return name_; }

   protected:
    Tensor &_eval(bool recompute = true) override;
    Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) override;

    std::string name_;
};

/** Returns a new variable operation. The variable wraps around val with name name.
 * @tparam T
 * @param name the name of the variable. This does not effect computation. It will allow to_string() methods to work,
 * however.
 * @param val tensor to wrap around
 * @return Variable<T>*
 */
inline Operation *var(std::string name, Tensor &val) { return default_graph.add_operation<Variable>(name, val); }

/** Returns a new variable operation. This version constructs a new tensor to store in the variable.
 * @tparam T
 * @param name
 * @param shape
 * @param filler
 * @param mem_type
 * @return Variable<T>*
 */
inline Operation *var(std::string name, const std::vector<index_t> &shape, DataType dtype, tensor_filler_t filler,
                      memory_t mem_type) {
    return default_graph.add_operation<Variable>(name, shape, dtype, filler, mem_type);
}

/** Creates a scalar variable.
 * @tparam T
 * @param name
 * @param val
 * @param mem_type
 * @return Variable<T>*
 */
template <typename T>
inline Operation *scalar(std::string name, T val, memory_t mem_type) {
    return default_graph.add_operation<Variable>(name, 1, ::magmadnn::GetDataType<T>::value,
                                                 {CONSTANT, static_cast<double>(val)}, mem_type);
}
}  // namespace op
}  // namespace magmadnn
