# skepsi

A neural network library in c++ aimed at providing a simple, modularized framework for deep learning. 

===== VERSION 0.0.1 =====
- Currently skepsi provides a dynamic memory manager, tensor wrapper for the memory manager, and a set of math operations for the tensor.
- More is coming...


### Download and Installation
-----------------------------

##### Dependencies
Skepsi uses `make` as its build system, so it must be installed on your system and in your path before you can build the library. The build also requires a c++11 capable compiler.

If compiling with GPU capabilities, then CUDA and likewise nvcc must be installed and in the proper PATHs. Skepsi has only been tested on Ubuntu (>16) and MacOS using CUDA (>9.0), however it is likely to work on most *nix based systems with a recent CUDA install. 

Skepsi makes heavy use of BLAS libraries. For Host only code a CBLAS library must be installed (such as [openblas](https://www.openblas.net/), [atlas](http://math-atlas.sourceforge.net/), etc...). If using the Device, then [Magma](http://icl.cs.utk.edu/magma/) (>2.5.0) must be installed.

##### Download
First get the repository on your computer with

```sh
git glone https://github.com/Dando18/skepsi
cd skepsi
```

##### Install
Next copy the make include settings into the head directory and edit them to your preferences.

```sh
cp make.inc-examples/make.inc-standard ./make.inc
vim make.inc # if you want to edit the settings
```

After this simply run `make install` to build and install skepsi. If your prefix (install location) has root priviledge access, then you'll need to run with `sudo`.

So the entire script looks like:

```sh
git clone https://github.com/Dando18/skepsi
cd skepsi
cp make.inc-examples/make.inc-standard ./make.inc
sudo make install
```

### Testing 
------------
Skepsi comes with some tester files to make sure everything is working properly. You can build them using the same makefile as for installation. Use the following commands to build and run the testers:

```sh
make testing
cd testing
sh run_tests.sh
```

### Task List (what's coming next):
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
- [ ] Basic Layer classes (Dense, Activation, Flatten, CNN)
- [ ] Model with forward/backward propagation
- [ ] Tests for Model/Layer training
- [ ] Parallel training (Multi-GPU)
- [ ] Tests for parallel training
- [x] Examples in Examples/ folder
- [ ] Tutorial / Presentation Slides
- [ ] I/O methods for Tensors
- [ ] Tests for tensor I/O
- [ ] Implement RNN
- [ ] Tests for RNN
- [ ] Compute graph optimizers/minimizers
- [ ] Hyperparameter Optimization tools
- [ ] Tests for hyperparameter optimization tools
- [ ] Package/Install configuration
- [ ] Tune compilation and runtime parameters to hardware
- [ ] Test on different hardwares (intel, amd, nvidia)
- [ ] OpenCL support (possibly? perhaps with different BLAS)


_author:_ Daniel Nichols
