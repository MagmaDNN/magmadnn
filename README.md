# skepsi

A neural network library in c++ aimed at providing a simple, modularized framework for deep learning. 

===== VERSION 0.0.1 =====
- Currently skepsi only works on single devices and only has a very limited memory manager and tensor class.
- More is coming...


### Download and Installation
-----------------------------

##### Dependencies
Skepsi uses `make` as its build system, so it must be installed on your system before you can build the library. The build also requires a c++11 capable compiler. If compiling with GPU capabilities, then CUDA and likewise nvcc must be installed and in the proper PATHs. Skepsi has only been tested on Ubuntu (>16) and MacOS with CUDA (>9.0), however it is likely to work on most *nix based systems with a proper CUDA install.

##### Download
First get the repository on your computer with

```sh
git glone https://github.com/Dando18/skepsi
cd skepsi
```

###### Install
Next copy the make include settings into the head directory and edit them to your preferences.

```sh
cp make.inc-examples/make.inc-standard ./make.inc
vim make.inc # if you want to edit the settings
```

After this simply run `make install` to build and install skepsi. If your prefix (install location) has root priviledge acccess, then you'll need to run with `sudo`.

So the entire script looks like,

```sh
git clone https://github.com/Dando18/skepsi
cd skepsi
cp make.inc-examples/make.inc-standard ./make.inc
sudo make install
```

### Testing 
------------
Skepsi comes with some tester files to make sure everything is working properly. You can build them using the same makefile as for installation. Use the following command to build and run the testers:

```sh
make testing
cd testing
sh ./run_tests.sh
```

### Task List (what's coming next):
-----------------------------------
- [x] Implement a memory manager for handling host, device, and managed memory
- [x] Build tests for memory manager
- [x] Implement Tensor wrapper around memory manager for multi-axis data storage
- [x] Build tests for tensor 
- [x] Create Make file and Build/Install system
- [x] Creates Docs and doxygen config
- [ ] Implement compute graph and basic operations for tensors
- [ ] Build tests for compute graph and tensor operations
- [ ] Link with BLAS/LAPACK (OpenBLAS?) and MAGMA
- [ ] Implement basic Layer classes (Dense, Activation, Flatten, CNN)
- [ ] Implement Model with forward/backward propagation
- [ ] Build tests for Model/Layer training
- [ ] Add examples in Examples/ folder
- [ ] Add I/O methods for Tensors
- [ ] Build tests for I/O
- [ ] Implement RNN
- [ ] Build tests for RNN
- [ ] Hyperparameter Optimization tools
- [ ] Build tests for hyperparameter optimization tools
- [ ] Package/Install configuration
- [ ] Tune compilation and runtime parameters to hardware
- [ ] Test on different hardwares (intel, amd, nvidia)
- [ ] OpenCL support (possibly? perhaps with different BLAS)


_author:_ Daniel Nichols
