## TODO List
-------------

The TODO section for this project is broken into course and fine grained tasks.

### Course Task List:
-----------------------------------
- [x] Memory manager for handling host, device, and managed memory
- [x] Tests for memory manager
- [x] Tensor wrapper around memory manager for multi-axis data storage
- [x] Tests for tensor 
- [x] Create Make file and Build/Install system
- [x] Creates Docs and doxygen config
- [x] Compute graph and basic operations for tensors
- [x] Tests for compute graph and tensor operations
- [x] Link with BLAS/LAPACK (OpenBLAS?) and MAGMA
- [x] Basic Layer classes (Dense, Activation, Flatten, CNN)
- [x] Model with forward/backward propagation
- [x] Tests for Model/Layer training
- [x] Optimizers
- [x] Tests for Optimizers
- [x] Parallel training (Multi-GPU)
- [ ] Tests for parallel training
- [x] Examples in Examples/ folder
- [x] Tutorial / Presentation Slides
- [x] Automatic or numerical gradient computations
- [x] Test gradient computations
- [x] I/O methods for Tensors
- [ ] Tests for tensor I/O
- [x] Batch Loaders
- [ ] Preprocessing methods (PCA, LDA, encoding)
- [ ] Tests for preprocessing methods
- [ ] Implement RNN
- [ ] Tests for RNN
- [ ] Compute graph optimizers/minimizers
- [ ] Hyperparameter Optimization tools
- [ ] Tests for hyperparameter optimization tools
- [ ] Package/Install configuration (deb packages, etc...)
- [ ] Tune compilation and runtime parameters to hardware
- [ ] Test on different hardwares (intel, amd, nvidia)
- [ ] OpenCL support (possibly? perhaps with different BLAS)
- [ ] AMD support (work with Frontier)

### Fine Task List:
-----------------------------------
- [x] Ensure CPU and GPU training results are the same.
- [ ] Revise memory system with compute graph and tensors. Check with gdb. Possibly replace operation references with smart pointers.
- [ ] remove unused operation _internal files.
- [x] check and fix speed of get/set with vector access
- [ ] tensor axis iterators
- [ ] CPU only convolution
- [ ] Fast ReduceSum
- [ ] Scalar Network output bug (CuDNN reduce sum issue)
