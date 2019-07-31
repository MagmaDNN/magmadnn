#pragma once

#include "compute/gcnconv/gcnconvop.h"
#include "layer/layer.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include <cmath>
#include <vector>

namespace magmadnn {
namespace layer {
//  layer for doing GCNConv as defined by Kipf and Welling in "Semi-Supervised Classification with Graph Convolutional
//  Networks" (arXiv:1609.02907)
//  input:           operation pointer, output shape batch_size * n_vertex * n_channelIn
//  output_channel:  number of output channels, n_channelOut
//  output shape:    batch_size * n_vertex * n_channelOut
/**
 * @brief layer for doing GCNConv as defined by Kipf and Welling in "Semi-Supervised Classification with Graph Convolutional Networks" (arXiv:1609.02907)
 * 
 * @tparam T
 */
template <typename T>
class KWGCNLayer : public Layer<T> {
   protected:
    Tensor<T>* weights_tensor;
    Tensor<T> laplacian;
    op::Operation<T>* weights;
    unsigned output_channel;
    unsigned V;
    bool copy;
    bool needs_grad;

   public:
    KWGCNLayer(op::Operation<T>* input, const Tensor<T>& adjacencyMatrix, unsigned output_channel, bool is_laplacian = false, bool check_graph = false, bool copy = true,
               bool needs_grad = true);
    virtual ~KWGCNLayer(void);
    inline op::Operation<T>* get_weight(void) { return weights; }
    std::vector<op::Operation<T>*> get_weights(void);
};

/**
 * @brief layer for doing GCNConv as defined by Kipf and Welling in "Semi-Supervised Classification with Graph Convolutional Networks" (arXiv:1609.02907)
 * 
 * @tparam T 
 * @param input Operation pointer, input from previous layers, shape {batch_size, n_vert, n_channel_in}
 * @param adjacencyMatrix Tensor pointer, adjacency matrix of the struct graph, shape {n_vert, n_vert}
 * @param output_channel Unsigned integer, number of output channels
 * @param is_laplacian Boolean, if false then adjacencyMatrix is a raw adjacency matrix, if true than adjacencyMatrix is the laplacian used in graph convolution
 * @param copy 
 * @param needs_grad 
 * @return KWGCNLayer<T>* Layer pointer, output shape {batch_size, n_vert, output_channel}
 */
template <typename T>
KWGCNLayer<T>* kwgcn(op::Operation<T>* input, const Tensor<T>& adjacencyMatrix, unsigned output_channel, bool is_laplacian = false, bool copy = true,
bool needs_grad = true);

}  // namespace layer
}  // namespace magmadnn
