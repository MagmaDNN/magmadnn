## Contributing
------------------
Below is a guideline on how to contribute to the project.
#### Table of Contents:
[Project Layout](#project-layout)

[Naming Conventions](#naming-conventions)

[GPU Code](#GPU-Code)

[Pull Requests](#pull-requests)


## Project Layout
-----------------
The project is split into several main subdirectories: [include](#include), [src](#src), [docs](#docs), [scripts](#scripts), [testing](#testing), [examples](#examples), and [make.inc-examples](#make.inc-examples).

#### include
`include` contains the header files for the entire project. `include` and `src` should have the same folder structure.
#### src
`src` contains the source files (.cpp and .cu) for the entire library. Everything defined in `include` must be implemented here.
#### docs
The `docs` folder contains these contributing guidelines and is the destination for doxygen output. Assuming doxygen is properly installed, running `make docs` (or `doxygen doxygen.config`) in the project root directory will build the docs into the `docs` folder.
#### scripts
The scripts directory contains several tools to assist in the development of MagmaDNN. More information is located in [scripts/README.md](/scripts/README.md).
#### testing
`testing` contains ~unit-test-ish type tests. They are intended to ensure that the library is installed and running correctly. If `sh run_tests.sh` does not succeed, then something went wrong with the installation.
#### examples
The examples are meant to be more like tutorial examples than the testing files. These contain more practical use cases for the library functionality.
#### make.inc-examples
Contains template make.inc files that can be used in the build and installation phase.

## Naming Conventions
---------------------
There are several different naming conventions used throughout the project. 

### File Naming
First and foremost, device and host distinction. Code that is device kernels only should be in their own source files. These source files should be named `*_device.cu`. Device files should only contain kernels and host functions to call these kernels.

Host code files do not require a special ending and should be in `*.cpp` files. 

Any internal utility functions should be in the `magmadnn::...::internal` namespace and should reside in files with a name such as `*_internal*.*`. For instance, a file that contains tensor addition utilities might be called `tensor_add_internal.cpp`. If that same utility contained device code, then it could be `tensor_add_internal_device.cu`. 

If a file contains a class, then the filename should be the same as the classname. For example, the class `foo` should reside in files `foo.h` and `foo.cpp`.

### Operations
Operations must be defined in `include/compute` and implemented in `src/compute`. Each operation should reside in its own folder. The operation class and filenames should be postfixed by `Op` (i.e. `matmulop.cpp`). All operations should lie in the `magmadnn::op` namespace. For more information on creating operations [see here](/include/compute/README.md).


## GPU Code
------------
MagmaDNN uses CUDA to power its GPU capabilities. To offer a more streamlined compile process all CUDA code in MagmaDNN is guarded by the macro `_HAS_CUDA_`. So code that utilizes any CUDA functionality should look similar to,

```c++
#if defined(_HAS_CUDA_)
cudaMalloc((void **)&ptr, n_bytes); /* or any other CUDA code */
#endif
```

Special CUDA syntax, such as kernel calls, must reside in `.cu` files, since nvcc is only used to compile `.cu` files.

### Supporting GPU operations
MagmaDNN aims to support both CPU and GPU operations. However, the focus of optimization is on CPU/GPU builds and not CPU-only. Thus it is imperative to implement an algorithm on the GPU that is optimized, and the CPU only version can be merely functional. While speed all around is ideal, the focus is on optimizing the CPU/GPU pairing.


## Pull Requests
----------------
