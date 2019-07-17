/**
 * @file operation.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-18
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <map>
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
    Operation() : has_been_computed(false) {}
    Operation(const std::vector<Operation<T> *> &inputs, bool needs_grad = true)
        : inputs(inputs), needs_grad(needs_grad) {
        for (typename std::vector<Operation<T> *>::iterator vit = inputs.begin(); vit != inputs.end(); vit++) {
            if (needs_grad) { /* TODO : verify this is necessary */
                (*vit)->add_consumer(this);
            }
        }
    }

    virtual Operation &operator=(const Operation &o) = delete;

    virtual ~Operation() {
        // for (unsigned int i = 0; i < inputs.size(); i++) delete inputs[i];

        /*  TODO : figure out why this peice of code caused SEGFAULTS
        if (this->output_tensor != NULL) {
            delete this->output_tensor;
        }
        */
    }

    /** Returns the expected output shape of this operation.
     * @deprecated since v1.2
     * @see shape()
     * @return std::vector<unsigned int>
     */
    virtual const std::vector<unsigned int> &get_output_shape() const { return this->output_shape; }

    /** returns the output shape of this operation
     * @return const std::vector<unsigned int>&
     */
    virtual const std::vector<unsigned int> &shape() const { return this->output_shape; }

    /**
     * @deprecated since v1.2
     * @see shape(unsigned int)
     * @param idx
     * @return std::vector<unsigned int>
     */
    virtual unsigned int get_output_shape(unsigned int idx) const { return this->output_shape.at(idx); }

    /** the size of axis idx of the output of this operation
     * @param idx
     * @return unsigned int
     */
    virtual unsigned int shape(unsigned int idx) const { return this->output_shape.at(idx); }

    /** The total number of elements outputted by operation.
     * @deprecated since v1.2
     * @see size()
     * @return unsigned int
     */
    virtual unsigned int get_output_size() const {
        unsigned int size = 1;
        for (unsigned int i = 0; i < this->output_shape.size(); i++) size *= this->output_shape[i];
        return size;
    }

    /** The total number of elements outputted by this operation
     * @return unsigned int
     */
    virtual unsigned int size() const {
        unsigned int size = 1;
        for (unsigned int i = 0; i < this->output_shape.size(); i++) size *= this->output_shape[i];
        return size;
    }

    /** The memory type used to compute this operation.
     * @return memory_t
     */
    virtual memory_t get_memory_type() const { return this->mem_type; }

    /** Returns the operation's evaluated tensor.
     * @param recompute whether to use previous value or recalculate
     * @return Tensor<T>*
     */
    virtual const Tensor<T> &eval(bool recompute = true) {
        if (!recompute && this->has_been_computed) {
            return this->output_tensor;
        } else {
            this->has_been_computed = true;
            return _eval(recompute);
        }
    }

    /** Clears the operation so that it will be recomputed.
     */
    virtual void reset() {
        this->has_been_computed = false;
        this->has_grad_been_computed = false;
    }

    /** Computes the gradient with respect to the outputs and var.
     * @param consumer the operation that consumes this that needs the gradient
     * @param grad the gradient of the loss w.r.t. the consumers output
     * @return Tensor<T>*
     */
    virtual const Tensor<T> &grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad,
                                  bool recompute = true) {
        if (!recompute) {
            return this->_grad_cache[var];
        } else {
            return _grad(consumer, var, grad);
        }
    }

    /**
     * @param consumer
     */
    virtual void add_consumer(Operation<T> *consumer) { this->consumers.push_back(consumer); }

    /** Returns a vector of operations that need this operation as input.
     * @return std::vector<Operation<T> *> vector of consumer operations
     */
    virtual const std::vector<Operation<T> *> &get_consumers() const { return this->consumers; }

    /** Returns a vector the input operations to this one.
     * @return std::vector<Operation<T> *> vector of input operations
     */
    virtual const std::vector<Operation<T> *> &get_inputs() const { return this->inputs; }

    /** Gets a pointer to the output tensor this returns
     * @return Tensor<T>*
     */
    virtual Tensor<T> &get_output_tensor() { return this->output_tensor; }

    virtual const Tensor<T> &get_output_tensor() const { return this->output_tensor; }

    /** Gets the current grad_tensor wrt to wrt.
     * @param wrt
     * @return Tensor<T>*
     */
    virtual const Tensor<T> &get_grad_tensor(Operation<T> *wrt) {
        /* TODO -- replace without grad_cache */
        return this->_grad_cache.find(wrt)->second;
    }

    /** string form of the given operation. Expands on input.
     * @return std::string
     */
    virtual std::string to_string() const = 0;

    /** the name of this operation.
     * @return std::string
     */
    virtual std::string get_name() const { return this->name; }

   protected:
    /** Sets this->output_tensor to the value of this operation
     * @return Tensor<T>* the evaluated tensor
     */
    virtual const Tensor<T> &_eval(bool recompute = true) = 0;

    /** Computes the gradient of this operation wrt the output of consumer.
     * @param consumer
     * @param var
     * @param grad
     * @param recompute
     * @return Tensor<T>*
     */
    virtual const Tensor<T> &_grad(Operation<T> *consumer, Operation<T> *var, const Tensor<T> &grad) = 0;

    std::vector<Operation<T> *> inputs;    /* children */
    std::vector<Operation<T> *> consumers; /* parents */

    std::vector<unsigned int> output_shape;
    memory_t mem_type;

    /* TODO -- get rid of _grad_cache */
    std::map<Operation<T> *, Tensor<T>> _grad_cache; /* this will cache the tensors for the gradient computation */
    std::string name = "DefaultOpName";

    Tensor<T> output_tensor; /* the return tensor */

    bool needs_grad;
    bool has_been_computed;
    bool has_grad_been_computed;
};

}  // namespace op
}  // namespace magmadnn
