## Troubleshooting
--------------------

#### Installation
  
- CUDA issues
  
  If MagmaDNN is building in CPU-only mode, but you're expecting it to compile for the GPU, then make sure the CUDA toolkit is in your path.
  Identify where your cuda installation is located and run `export PATH="${PATH}:/path/to/cuda/bin"` in your shell.
   
