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
//  struct_graph:    graph object, of order n_vertex, in sparse format
//  output_channel:  number of output channels, n_channelOut
//  output shape:    batch_size * n_vertex * n_channelOut
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

template <typename T>
KWGCNLayer<T>* kwgcn(op::Operation<T>* input, const Tensor<T>& adjacencyMatrix, unsigned output_channel, bool is_laplacian = false, bool copy = true,
bool needs_grad = true);

}  // namespace layer
}  // namespace magmadnn
