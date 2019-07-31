# MagmaDNN GCN

This repo is forked from the C++ neural network library [MagmaDNN](https://github.com/MagmaDNN/magmadnn) (commit hash [8e80c066ee52aa07b0f73eb0e9e55e22fcb857a2](https://github.com/MagmaDNN/magmadnn/commit/8e80c066ee52aa07b0f73eb0e9e55e22fcb857a2)). 

In this repo, a graph convolution layer (or to be more specific the one developed by [Kipf & Welling](https://arxiv.org/abs/1609.02907)) is implemented. The GPU implementation relies on CUDA CuBlas library. 

At the moment the implementation uses dense matrix multiplications in batches and hence supports only graphs in dense format. Sparse graphs may be supported if Magma/CuSparse/other libraries have the corresponding routines or native routines are implemented. 

Other graph convolution layers and pooling layers may be added in the future. 

As this is forked from the dev branch of MagmaDNN, bugs may exist and features in this repo may be removed in upcoming MagmaDNN versions. 

_author:_ Kam Fai Chan

_original author:_ Daniel Nichols

_original co-author:_ Sedrick Keh
