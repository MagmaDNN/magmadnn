# if (MAGMADNN_ENABLE_CUDA)
#   cuda_add_library(magmadnn "")
# else()
if (MAGMADNN_BUILD_SHARED_LIBS)
  add_library(magmadnn SHARED "")
else()
  add_library(magmadnn STATIC "")
endif()
# endif()

target_sources(magmadnn
  PRIVATE
  init_finalize.cpp
  exception.cpp
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

if (MAGMADNN_ENABLE_CUDA)
  target_sources(magmadnn
    PRIVATE
    compute/add/geadd_internal_device.cu
    compute/crossentropy/crossentropy_internal_device.cu
    compute/div/div_internal_device.cu
    compute/log/log_internal_device.cu
    compute/negative/negative_internal_device.cu
    compute/pow/pow_internal_device.cu
    compute/product/product_internal_device.cu
    compute/reducesum/reducesum_internal_device.cu
    compute/relu/relu_internal_device.cu
    compute/scalarproduct/scalarproduct_internal_device.cu
    compute/sigmoid/sigmoid_internal_device.cu
    compute/sum/sum_internal_device.cu
    compute/tanh/tanh_internal_device.cu
    compute/transpose/transpose_internal_device.cu
    )
endif()

# data
target_sources(magmadnn
  PRIVATE
  data/utils.cpp
  data/MNIST.cpp
  data/CIFAR10.cpp
  data/CIFAR100.cpp)

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

if (MAGMADNN_ENABLE_CUDA)
  target_sources(magmadnn
    PRIVATE
    math/bias_add_device.cu
    math/crossentropy_device.cu
    math/optimizer_math/adagrad_device.cu
    math/optimizer_math/adam_device.cu
    math/optimizer_math/rmsprop_device.cu
    math/optimizer_math/sgd_momentum_device.cu
    math/pow_device.cu
    math/scalar_tensor_product_device.cu
    math/sum_device.cu)
endif ()

# memory
target_sources(magmadnn
  PRIVATE
  memory/memorymanager.cpp
  )

if (MAGMADNN_ENABLE_CUDA)
  target_sources(magmadnn
    PRIVATE
    memory/memory_internal_device.cu)
endif ()
    
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

if (MAGMADNN_ENABLE_CUDA)
  target_sources(magmadnn
    PRIVATE
    optimizer/gradientdescent/gradientdescent_internal_device.cu)
endif ()

# tensor
target_sources(magmadnn
  PRIVATE
  tensor/fill_internal_host.cpp
  tensor/tensor.cpp
  tensor/tensor_internal.cpp
  tensor/tensor_io.cpp)

if (MAGMADNN_ENABLE_CUDA)
  target_sources(magmadnn
    PRIVATE
    tensor/fill_internal_device.cu)
endif ()

magmadnn_compile_features(magmadnn)
magmadnn_default_includes(magmadnn)

magmadnn_install_library(magmadnn)

# if (MAGMADNN_ENABLE_CUDA)
# magmadnn_compile_features(magmadnn_cuda)
# magmadnn_default_includes(magmadnn_cuda)

# magmadnn_install_library(magmadnn_cuda)
# endif ()
