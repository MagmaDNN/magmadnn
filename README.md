# MagmaDNN

A neural network library in c++ aimed at providing a simple, modularized framework for deep learning. This is a Work-In-Progress replacement for [MagmaDNN](https://bitbucket.org/icl/magmadnn).

===== VERSION 0.1.0 =====
- Currently MagmaDNN provides a dynamic memory manager, tensor wrapper for the memory manager, and a set of math operations for the tensor.
- As of 0.1.0 it now has support for a full compute graph, gradient computation, forward/backward propagation, and basic NN Layers.

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
