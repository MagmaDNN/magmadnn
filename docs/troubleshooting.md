## Troubleshooting
--------------------

#### Installation
  
- CUDA issues
  
  If MagmaDNN is building in CPU-only mode, but you're expecting it to compile for the GPU, then make sure the CUDA toolkit is in your path.
  Identify where your cuda installation is located and run `export PATH="${PATH}:/path/to/cuda/bin"` in your shell.

  'Cannot find header "cudnn.h"' or similar errors. Make sure you have CuDNN installed. See [this site](https://developer.nvidia.com/cudnn) for more info on installing CuDNN.

  'Cannot find header "magma.h"' or similar errors. Magma must be installed in order to use MagmaDNN. See [this site](http://icl.cs.utk.edu/magma/) for more info on installing Magma.
   



- Runtime Errors

    Stack smashing detected -- Currently there is a known memory bug in MagmaDNN revolving around the `::magmadnn::model::NeuralNetwork<T>::fit` function. If you pass stack allocated tensor objects as the `x` and `y` (i.e. `Tensor<float> x (...);`), then somewhere in the network training, the stack is accessed out-of-bounds. MagmaDNN 0.1.1 or 0.1.2 aims to have an overhauled memory system, which handles this issue. As a workaround, please pass heap allocated tensor objects to the fit function (i.e. `Tensor<float> *x = new Tensor<float> (...);`).