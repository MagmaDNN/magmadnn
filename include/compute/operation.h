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

namespace magmadnn {
namespace op {

template <typename T>
class Operation {
public: 
    /** The operation class serves as an abstract object, which all tensors operations descend
     *  from. It is used to build a computation tree.
     */
    Operation() {}
    Operation(std::vector<Operation<T> *> inputs) : inputs(inputs) {
        for (typename std::vector<Operation<T> *>::iterator vit = inputs.begin(); vit != inputs.end(); vit++) {
            (*vit)->add_consumer(this);
        }
    }
	virtual ~Operation() {
        for (unsigned int i = 0; i < inputs.size(); i++)
            delete inputs[i];
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

    /** Computes the gradient with respect to the outputs and var.
     * @param consumer the operation that consumes this that needs the gradient
     * @param grad the gradient of the loss w.r.t. the consumers output
     * @return Tensor<T>* 
     */
    virtual Operation<T>* grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) = 0;

    /**
     * @param consumer 
     */
    virtual void add_consumer(Operation<T> *consumer) { this->consumers.push_back(consumer); }

    /** Returns a vector of operations that need this operation as input.
     * @return std::vector<Operation<T> *> vector of consumer operations
     */
    virtual std::vector<Operation<T> *> get_consumers() { return this->consumers; }

    /** Returns a vector the input operations to this one.
     * @return std::vector<Operation<T> *> vector of input operations
     */
    virtual std::vector<Operation<T> *> get_inputs() { return this->inputs; }

    /** string form of the given operation. Expands on input.
     * @return std::string 
     */
    virtual std::string to_string() = 0;
    
protected:
    std::vector<Operation<T>*> inputs;
    std::vector<Operation<T>*> consumers;
    std::vector<unsigned int> output_shape;
    memory_t mem_type;

    bool needs_grad;
};

} // namespace op
} // namespace magmadnn
