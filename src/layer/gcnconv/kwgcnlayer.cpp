#include "layer/gcnconv/kwgcnlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
KWGCNLayer<T>::KWGCNLayer(op::Operation<T>* input, const Tensor<T>& adjacencyMatrix, unsigned output_channel,
                          bool is_laplacian, bool check_graph, bool copy, bool needs_grad)
    : Layer<T>(input->get_output_shape(), input),
      laplacian(Tensor<T>(adjacencyMatrix.get_shape(), input->get_memory_type())),
      output_channel(output_channel),
      copy(copy),
      needs_grad(needs_grad) {
    assert(T_IS_MATRIX(&adjacencyMatrix));
    assert(adjacencyMatrix.get_shape(0) == adjacencyMatrix.get_shape(1));
    assert(this->get_input_shape().size() == 3);
    assert(adjacencyMatrix.get_shape(0) == this->get_input_shape(1));
    this->name = "KWGCNLayer / " + std::to_string(output_channel);
    T bound = static_cast<T>(sqrt(2.0 / this->input->get_output_shape(2)));
    this->weights_tensor = new Tensor<T>({this->input->get_output_shape(2), this->output_channel},
                                         {UNIFORM, {-bound, bound}}, this->input->get_memory_type());
    this->weights = op::var("__" + this->name + "_layer_weights", this->weights_tensor);
    V = adjacencyMatrix.get_shape(0);
    laplacian.copy_from(adjacencyMatrix);
    if (check_graph) {
        for (unsigned i = 0; i < V - 1; ++i) {
            if (!is_laplacian) {
                assert(adjacencyMatrix.get({i, i}) == 0 && "Self-loop detected for input graph in KWGCNLayer.\n");
            }
            for (unsigned j = i + 1; j < V; ++j) {
                assert(adjacencyMatrix.get({i, j}) == adjacencyMatrix.get({j, i}) &&
                       "Input graph in KWGCNLayer must be undirected");
                assert(adjacencyMatrix.get({i, j}) >= 0 && "Input graph in KWGCNLayer must not have negative weight");
            }
        }
    }
    if (!is_laplacian) {
        Tensor<T> Dtilde({V, 1}, {NONE, {}}, DEVICE);
        Tensor<T> ones({V, 1}, {ONE, {}}, DEVICE);  //  to do: check if matmul is inplace
        math::matmul<T>(1, false, &laplacian, false, &ones, 0, &Dtilde);
        T A, newA;
        for (unsigned i = 0; i < V; ++i) {
            for (unsigned j = 0; j < V; ++j) {
                A = laplacian.get({i, j});
                if (i == j) {
                    newA = (A == 0 ? 1 : A) / Dtilde.get(i);
                } else {
                    newA = (A == 0 ? (T) 0 : (A + (T) 1) / std::sqrt(Dtilde.get(i) * Dtilde.get(j)));
                }
                if (A != newA) {
                    laplacian.set({i, j}, newA);
                }
            }
        }
    }
    this->output = op::gcnconvop(&this->laplacian, this->input, this->weights);
}
template <typename T>
KWGCNLayer<T>::~KWGCNLayer(void) {
    delete weights_tensor;
}
template <typename T>
std::vector<op::Operation<T>*> KWGCNLayer<T>::get_weights(void) {
    return {this->weights};
}
template class KWGCNLayer<int>;
template class KWGCNLayer<float>;
template class KWGCNLayer<double>;

template <typename T>
KWGCNLayer<T>* kwgcn(op::Operation<T>* input, const Tensor<T>& adjacencyMatrix, unsigned output_channel,
                     bool is_laplacian, bool copy, bool needs_grad) {
    return new KWGCNLayer<T>(input, adjacencyMatrix, output_channel, is_laplacian, true, copy, needs_grad);
}
template KWGCNLayer<int>* kwgcn(op::Operation<int>* input, const Tensor<int>& adjacencyMatrix, unsigned output_channel,
                                bool is_laplacian, bool copy, bool needs_grad);
template KWGCNLayer<float>* kwgcn(op::Operation<float>* input, const Tensor<float>& adjacencyMatrix,
                                  unsigned output_channel, bool is_laplacian, bool copy, bool needs_grad);
template KWGCNLayer<double>* kwgcn(op::Operation<double>* input, const Tensor<double>& adjacencyMatrix,
                                   unsigned output_channel, bool is_laplacian, bool copy, bool needs_grad);

}  // namespace layer
}  // namespace magmadnn
