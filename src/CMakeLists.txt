add_library(magmadnn "")

target_sources(magmadnn
  PRIVATE
  init_finalize.cpp
  utilities_internal.cpp)

# compute
target_sources(magmadnn
  PRIVATE
  compute/add/addop.cpp
  compute/add/geadd_internal.cpp
  compute/batchnorm/batchnormop.cpp
  compute/crossentropy/crossentropy_internal.cpp
  compute/conv2dforward/conv2dforwardop.cpp
  compute/crossentropy/crossentropy_internal.cpp
  compute/crossentropy/crossentropyop.cpp
  compute/div/div_internal.cpp
  compute/div/divop.cpp
  compute/dot/dotop.cpp
  compute/dropout/dropoutop.cpp
  compute/flatten/flattenop.cpp
  compute/gradients.cpp
  compute/gradtable.cpp  
  compute/linearforward/linearforwardop.cpp
  compute/log/log_internal.cpp
  compute/log/logop.cpp
  compute/matmul/gemm_internal.cpp
  compute/matmul/matmulop.cpp
  compute/meansquarederror/meansquarederror.cpp
  compute/negative/negative_internal.cpp
  compute/negative/negativeop.cpp
  compute/op_utilities.cpp
  compute/pooling/poolingop.cpp
  compute/pow/pow_internal.cpp
  compute/pow/powop.cpp
  compute/product/product_internal.cpp
  compute/product/productop.cpp
  compute/reducesum/reducesum_internal.cpp
  compute/reducesum/reducesumop.cpp
  compute/relu/relu_internal.cpp
  compute/relu/reluop.cpp
  compute/scalarproduct/scalarproduct_internal.cpp
  compute/scalarproduct/scalarproductop.cpp
  compute/sigmoid/sigmoid_internal.cpp
  compute/sigmoid/sigmoid_op.cpp
  compute/softmax/softmaxop.cpp
  compute/sum/sum_internal.cpp
  compute/sum/sumop.cpp
  compute/tanh/tanh_internal.cpp
  compute/tanh/tanhop.cpp
  compute/transpose/transpose_internal.cpp
  compute/transpose/transposeop.cpp
  compute/variable.cpp)

# dataloader
target_sources(magmadnn
  PRIVATE
  dataloader/linear/linearloader.cpp)

# layer
target_sources(magmadnn
  PRIVATE
  layer/activation/activationlayer.cpp
  layer/batchnorm/batchnormlayer.cpp
  layer/conv2d/conv2dlayer.cpp
  layer/dropout/dropoutlayer.cpp
  layer/flatten/flattenlayer.cpp
  layer/fullyconnected/fullyconnectedlayer.cpp
  layer/input/inputlayer.cpp
  layer/layer_utilities.cpp
  layer/output/outputlayer.cpp
  layer/pooling/poolinglayer.cpp
  layer/wrapper.cpp)

# math
target_sources(magmadnn
  PRIVATE
  math/add.cpp
  math/argmax.cpp
  math/batchnorm.cpp
  math/bias_add.cpp
  math/concat.cpp
  math/conv2d.cpp
  math/crossentropy.cpp
  math/dot.cpp
  math/dropout.cpp
  math/matmul.cpp
  math/negate.cpp
  math/optimizer_math/adagrad.cpp
  math/optimizer_math/adam.cpp
  math/optimizer_math/rmsprop.cpp
  math/optimizer_math/sgd_momentum.cpp
  math/pooling.cpp
  math/pow.cpp
  math/product.cpp
  math/reduce_sum.cpp
  math/relu.cpp
  math/scalar_tensor_product.cpp
  math/softmax.cpp
  math/sum.cpp
  math/tile.cpp
  math/wrappers.cpp)

# memory
target_sources(magmadnn
  PRIVATE
  memory/memorymanager.cpp
  )

# model
target_sources(magmadnn
  PRIVATE
  model/neuralnetwork/neuralnetwork.cpp
  model/neuralnetwork/neuralnetwork_utilities.cpp
  )

# optimizer
target_sources(magmadnn
  PRIVATE
 optimizer/adagrad/adagrad.cpp
 optimizer/adam/adam.cpp
 optimizer/distributed_momentum_sgd.cpp
 optimizer/gradientdescent/gradientdescent.cpp
 optimizer/gradientdescent/gradientdescent_internal.cpp
 optimizer/rmsprop/rmsprop.cpp)

# tensor
target_sources(magmadnn
  PRIVATE
  tensor/fill_internal_host.cpp
  tensor/tensor.cpp
  tensor/tensor_internal.cpp
  tensor/tensor_io.cpp)

magmadnn_compile_features(magmadnn)
magmadnn_default_includes(magmadnn)
