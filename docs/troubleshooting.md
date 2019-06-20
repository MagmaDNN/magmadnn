## Troubleshooting
--------------------

#### Installation
  
- CUDA issues
  
  If MagmaDNN is building in CPU-only mode, but you're expecting it to compile for the GPU, then make sure the CUDA toolkit is in your path.
  Identify where your cuda installation is located and run `export PATH="${PATH}:/path/to/cuda/bin"` in your shell.

  'Cannot find header "cudnn.h"' or similar errors. Make sure you have CuDNN installed. See [this site](https://developer.nvidia.com/cudnn) for more info on installing CuDNN.

  'Cannot find header "magma.h"' or similar errors. Magma must be installed in order to use MagmaDNN. See [this site](http://icl.cs.utk.edu/magma/) for more info on installing Magma.
   
