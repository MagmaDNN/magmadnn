/**
 * @file layer.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <string>
#include <vector>
#include "compute/operation.h"

namespace magmadnn {
namespace layer {

template <typename T>
class Layer {
   public:
    virtual ~Layer(){};
    virtual std::vector<op::Operation<T> *> get_weights() = 0;

    virtual op::Operation<T> *out() { return output; }

    /** Get the number of parameters for this layer
     * @return unsigned int
     */
    virtual unsigned int get_num_params() { return 0; }

    /** Get the pointer to the input tensor for this layer
     * @return tensor<T>*
     */
    op::Operation<T> *get_input() { return input; }
    /** Get the pointer to the output tensor for this layer
     * @return tensor<T>*
     */
    op::Operation<T> *get_output() { return output; }

    /** Returns a copy of the input shape as a vector
     * @return std::vector<unsigned int>
     */
    std::vector<unsigned int> get_input_shape() const { return input_shape; }
    /** Returns a copy of the output shape as a vector
     * @return std::vector<unsigned int>
     */
    std::vector<unsigned int> get_output_shape() const { return this->output->get_output_shape(); }

    /** Gets the size of the i-th axis of the input tensor
     * @param i axis
     * @return std::vector<unsigned int>
     */
    unsigned int get_input_shape(unsigned int i) const {
        assert(i < input_shape.size());
        return input_shape[i];
    }

    /** Gets the size of the i-th axis of the output tensor
     * @param i axis
     * @return std::vector<unsigned int>
     */
    unsigned int get_output_shape(unsigned int i) const {
        assert(i < output_shape.size());
        return output_shape[i];
    }

    /** Set the name of this layer.
     * @param name name to assign to this layer
     */
    void set_name(std::string name) { this->name = name; }

    /** Returns the name of this layer. Defaults to the layer type if not set
     * on its own.
     * @return std::string the name of this layer.
     */
    std::string get_name() const { return this->name; }

    // Return an estimate of the memory needed to store the data for
    // this layer
    std::size_t get_memory_size() const { return 0; }

   protected:
    Layer(std::vector<unsigned int> input_shape, op::Operation<T> *input) : input_shape(input_shape), input(input) {}

    std::vector<unsigned int> input_shape;
    std::vector<unsigned int> output_shape;

    op::Operation<T> *input;
    op::Operation<T> *output;

    std::string name;
};

}  // namespace layer
}  // namespace magmadnn
