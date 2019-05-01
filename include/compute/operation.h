/**
 * @file operation.h
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-18
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include "tensor/tensor.h"

namespace skepsi {
namespace op {

template <typename T>
class Operation {
public: 
    /** The operation class serves as an abstract object, which all tensors operations descend
     *  from. It is used to build a computation tree.
     */
    Operation() {}
    Operation(std::vector<Operation<T>*> children) : children(children) {}
	virtual ~Operation() {
        for (unsigned int i = 0; i < children.size(); i++)
            delete children[i];
    }

    /** Returns the expected output shape of this operation.
     * @return std::vector<unsigned int> 
     */
    virtual std::vector<unsigned int> get_output_shape() const { return this->output_shape; }

    /**
     * @param idx 
     * @return std::vector<unsigned int> 
     */
    virtual unsigned int get_output_shape(unsigned int idx) const {
        assert( idx < this->output_shape.size() );
        return this->output_shape[idx];
    }

    /** The total number of elements outputted by operation.
     * @return unsigned int 
     */
    virtual unsigned int get_output_size() const {
        unsigned int size = 1;
        for (unsigned int i = 0; i < this->output_shape.size(); i++) size *= this->output_shape[i];
        return size;
    }

    /** The memory type used to compute this operation.
     * @return memory_t 
     */
    virtual memory_t get_memory_type() const { return this->mem_type; }

    /** Returns the operation's evaluated tensor.
     * @return Tensor<T>* 
     */
    virtual Tensor<T>* eval() = 0;

    /** Returns the gradient of this tensor with respect to its output.
     * @return Tensor<T>* 
     */
    virtual Tensor<T>* grad() = 0;

    /** string form of the given operation. Expands on children.
     * @return std::string 
     */
    virtual std::string to_string() = 0;
    
protected:
    std::vector<Operation<T>*> children;
    std::vector<unsigned int> output_shape;
    memory_t mem_type;

    bool needs_grad;
};

} // namespace op
} // namespace skepsi
