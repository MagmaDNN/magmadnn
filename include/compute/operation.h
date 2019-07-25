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

class Operation {
   public:
    /** The operation class serves as an abstract object, which all tensors operations descend
     *  from. It is used to build a computation tree.
     */
    Operation() : has_been_computed_(false) {}
    Operation(const std::vector<Operation *> &inputs, bool needs_grad = true)
        : inputs_(inputs), needs_grad_(needs_grad) {
        for (const auto &child : inputs) {
            if (needs_grad_) { /* TODO -- test this more (not completely sure why it's necessary) */
                child->add_consumer(this);
            }
        }
    }

    virtual Operation &operator=(const Operation &o) = delete;

    /** Returns the expected output shape of this operation.
     * @deprecated since v1.2
     * @see shape()
     * @return std::vector<unsigned int>
     */
    virtual const std::vector<index_t> &get_output_shape() const { return this->output_shape_; }

    /** returns the output shape of this operation
     * @return const std::vector<unsigned int>&
     */
    virtual const std::vector<index_t> &shape() const { return this->output_shape_; }

    /**
     * @deprecated since v1.2
     * @see shape(unsigned int)
     * @param idx
     * @return std::vector<unsigned int>
     */
    virtual index_t get_output_shape(index_t idx) const { return this->output_shape_.at(idx); }

    /** the size of axis idx of the output of this operation
     * @param idx
     * @return unsigned int
     */
    virtual index_t shape(index_t idx) const { return this->output_shape_.at(idx); }

    /** The total number of elements outputted by operation.
     * @deprecated since v1.2
     * @see size()
     * @return unsigned int
     */
    virtual size_t get_output_size() const { return this->size(); }

    /** The total number of elements outputted by this operation
     * @return unsigned int
     */
    virtual size_t size() const {
        return (size_t) std::accumulate(this->output_shape_.begin(), this->output_shape_.end(), static_cast<index_t>(1),
                                        std::multiplies<index_t>());
    }

    /** The memory type used to compute this operation.
     * @return memory_t
     */
    virtual memory_t get_memory_type() const { return this->mem_type_; }

    /** Returns the operation's evaluated tensor.
     * @param recompute whether to use previous value or recalculate
     * @return Tensor<T>*
     */
    virtual Tensor &eval(bool recompute = true) {
        if (!recompute && this->has_been_computed_) {
            return this->output_tensor_;
        } else {
            this->has_been_computed_ = true;
            return _eval(recompute);
        }
    }

    /** Clears the operation so that it will be recomputed.
     */
    virtual void reset() {
        this->has_been_computed_ = false;
        this->has_grad_been_computed_ = false;
    }

    /** Computes the gradient with respect to the outputs and var.
     * @param consumer the operation that consumes this that needs the gradient
     * @param grad the gradient of the loss w.r.t. the consumers output
     * @return Tensor<T>*
     */
    virtual Tensor &grad(Operation *consumer, Operation *var, const Tensor &grad, bool recompute = true) {
        if (!recompute) {
            return this->_grad_cache[var];
        } else {
            return _grad(consumer, var, grad);
        }
    }

    /**
     * @param consumer
     */
    virtual void add_consumer(Operation *consumer) { this->consumers_.push_back(consumer); }

    /** Returns a vector of operations that need this operation as input.
     * @return std::vector<Operation<T> *> vector of consumer operations
     */
    virtual const std::vector<Operation *> &get_consumers() const { return this->consumers_; }

    /** Returns a vector the input operations to this one.
     * @return std::vector<Operation<T> *> vector of input operations
     */
    virtual const std::vector<Operation *> &get_inputs() const { return this->inputs_; }

    /** Gets a pointer to the output tensor this returns
     * @return Tensor<T>*
     */
    virtual Tensor &get_output_tensor() { return this->output_tensor_; }

    virtual const Tensor &get_output_tensor() const { return this->output_tensor_; }

    /** Gets the current grad_tensor wrt to wrt.
     * @param wrt
     * @return Tensor<T>*
     */
    virtual const Tensor &get_grad_tensor(Operation *wrt) {
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
    virtual std::string get_name() const { return this->name_; }

   protected:
    /** Sets this->output_tensor to the value of this operation
     * @return Tensor<T>* the evaluated tensor
     */
    virtual Tensor &_eval(bool recompute = true) = 0;

    /** Computes the gradient of this operation wrt the output of consumer.
     * @param consumer
     * @param var
     * @param grad
     * @param recompute
     * @return Tensor<T>*
     */
    virtual Tensor &_grad(Operation *consumer, Operation *var, const Tensor &grad) = 0;

    inline void use_tensor_settings(const Tensor &t, bool use_shape = true) {
        this->mem_type_ = t.get_memory_type();
        this->dtype_ = t.dtype();
        if (use_shape) {
            this->output_shape_ = t.shape();
        }
    }

    std::vector<Operation *> inputs_;    /* children */
    std::vector<Operation *> consumers_; /* parents */

    std::vector<index_t> output_shape_;
    memory_t mem_type_;
    DataType dtype_;

    /* TODO -- get rid of _grad_cache */
    std::map<Operation *, std::reference_wrapper<Tensor>>
        _grad_cache; /* this will cache the tensors for the gradient computation */
    std::string name_ = "DefaultOpName";

    Tensor output_tensor_; /* the return tensor */

    bool needs_grad_;
    bool has_been_computed_;
    bool has_grad_been_computed_;
};

}  // namespace op
}  // namespace magmadnn
