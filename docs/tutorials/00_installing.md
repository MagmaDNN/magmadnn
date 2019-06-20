## Tutorial 0: Installing
-------------------------

#### Dependencies
First make sure you have the right set up to install MagmaDNN. It is recommended to use MagmaDNN on a modern CPU paired with a recent GPU. It is also recommended that you have >=8GB of memory. Below are the listed software requirements:

Any MagmaDNN install:

- c++11 capable compiler (recommended g++ >=6)
- BLAS implementation (openblas, atlas, intel-mkl, etc...)
- git
- make

MagmaDNN-GPU install:

- CUDA >=9.2
- CuDNN >=6.0
- Magma >=2.5.0

_Note:_ if installing dependencies in non-standard paths make sure to update `LD_LIBRARY_PATH` so that your operating system can find the shared object files.

#### Downloading

Currently MagmaDNN does not offer any pre-compiled binaries, so it must be built from source. Once you have installed all of the dependencies the source can be downloaded using:

```sh
git clone https://github.com/MagmaDNN/magmadnn
cd magmadnn
```

#### Compiling and Installing

MagmaDNN uses a `make.inc` file to set compile flags/options. The `make.inc-examples` directory contains examples. Copy one over and adjust your compile settings accordingly. The example make.inc files document which compile settings can be set.

```sh
cp ./make.inc-examples/make.inc-standard ./make.inc
vim make.inc  # set make settings
# or `nano make.inc` if you don't have vim
```

Once you have set up the `make.inc` file it is time to compile and install using

```sh
make install
```

The MagmaDNN compilation can take a couple minutes especially if compiling the GPU version. To speed it up compile in parallel with `make install -j4` (or `-jn` where _n_ is the number of cores in your CPU).


#### Testing
It is good to test your install to make sure everything is working. MagmaDNN comes with a suite of testers that will ensure your install is working correctly. To run them,

```sh
make testing
cd testing
sh run_tests.sh
```

#### Docs
MagmaDNN uses doxygen for its documentation. To build the docs you must have `doxygen` installed. If you want to make the reference manual pdf, then you must also have latex installed. To install them on ubuntu use `sudo apt install doxygen texlive-full`. To make the docs,

```sh
make docs
```

#### Examples
There are several example files in the `examples/` folder. They are simple and commented to give an idea for what MagmaDNN code typically looks like. They can be made with

```sh
make examples
```