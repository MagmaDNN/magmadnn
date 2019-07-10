# MagmaDNN

A neural network library in c++ aimed at providing a simple, modularized framework for deep learning that is accelerated for heterogeneous architectures. MagmaDNN's releases are located at [https://bitbucket.org/icl/magmadnn](https://bitbucket.org/icl/magmadnn) (and [here](https://icl.cs.utk.edu/magma/)), while active development occurs at [https://github.com/MagmaDNN/magmadnn/tree/dev](https://github.com/MagmaDNN/magmadnn/tree/dev). If you're looking to contribute or submit a pull-requests/issues, then please do so on the github development repository.

===== VERSION 1.0 =====

In version 1.0 MagmaDNN offers a strong tensor core with standard machine learning and DNN functionalities built around it. For nightly development builds use the github repository linked above.

MagmaDNN is optimized towards heterogeneous architectures (multi-core CPU and GPU), so it is advised to use with a modern NVIDIA GPU. However, MagmaDNN does support a CPU only install. This is mainly meant for testing and is not nearly as optimized as the GPU version.


The documentation can be found on the [docs site](https://magmadnn.github.io/magmadnn/html). For the most recent version of the documentation see the [build & install tutorial](/docs/tutorials/00_installing.md) on how to build the docs from source. The [todo page](/docs/todo.md) contains information on the future of the package and the [troubleshooting page](/docs/troubleshooting.md) walks through common issues and there solution.


### Tutorials
-------------
There are several tutorials in [docs/tutorials](/docs/tutorials). These give an introduction into installing and using the library.


### Examples
-----------
For examples of what MagmaDNN code looks like see the [examples/ folder](/examples). If MagmaDNN is downloaded and installed, then the examples can be made and run with `make examples`.


_author:_ Daniel Nichols

_co-author:_ Sedrick Keh
