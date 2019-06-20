## Tutorial 02: Tensors
-----------------------

Tensors are fundamental to machine learning operations. They store multi-dimensional data and are the basis for all the math operations in MagmaDNN.


MagmaDNN defines a `Tensor` class which stores and keeps track of tensors for us. To create one, simply use:

```c++
/* create a  10x5  tensor */
Tensor<float> example_tensor ({10,5});
```

The above creates a 10x5 tensor called `example_tensor`. The type inside the angle brackets dictates which precision is used. Tensors support `int`, `float`, and `double` precisions. The constructor for tensors allows for more customization. The full constructor is defined

```c++
Tensor<T>::Tensor(
    std::vector<unsigned int> shape,    /* vector of integer indices representing the shape */
    ::magmadnn::tensor_filler_t filler=DEFAULT_TENSOR_FILLER,   /* method to fill the tensor */
    ::magmadnn::memory_t memory_type=HOST,  /* what type of memory will this data be stored in */
    ::magmadnn::device_t device_id=0    /* which device will be used to calculate on this data */
    );
```

`tensor_filler_t` is a struct that defines how to initialize the data inside the tensor. There are several available options:

```c++
Tensor<float> none ({10}, {NONE,{}});   /* don't init. the data */
Tensor<float> constant ({10,5}, {CONSTANT, {5.0f}});    /* initialize the values to the constant 5.0f */
Tensor<float> ones ({3,2,10}, {ONE, {}});    /* initialize the values to 1.0f */
Tensor<float> zeros ({100}, {ZERO, {}});    /* initialize the values to zero */
Tensor<float> identity ({5,5}, {IDENTITY, {}});    /* initialize to the Identity matrix -- must be a square matrix */
Tensor<float> diagonal ({3,3}, {DIAGONAL, {1.0f, -1.0f, 3.14f}});   /* set the tensors diagonal elements to 1,-1,3.14. You can also pass one value which will be places across the diagonal. */
Tensor<float> uniform ({20}, {UNIFORM, {-1.0f, 1.0f}}); /* initialize to uniform values between -1 and 1 */
Tensor<float> normal ({5,4}, {GLOROT, {0.0f, 1.0f}});   /* normal distribution with mean 0.0 and std. dev. 1.0 */
```

`memory_t` is an enumerate that describes how data is stored within a tensor:

| Memory Type Name | Description                                                          | Needs Sync | MagmaDNN version  |
|------------------|----------------------------------------------------------------------|------------|-------------------|
| HOST             | CPU memory                                                           |     no     | MagmaDNN-CPU only |
| DEVICE           | GPU memory                                                           |     no     | MagmaDNN-GPU      |
| MANAGED          | Managed stores a copy on the  GPU and CPU.                           |     yes    | MagmaDNN-GPU      |
| CUDA_MANAGED     | Cuda_Managed uses CUDA's unified memory to keep data on CPU and GPU. |     yes    | MagmaDNN-GPU      |

For instance, the following code creates a 5x3 tensor filled with ones and stores it on the GPU.

```c++
Tensor<float> x ({5,3}, {ONE,{}}, DEVICE);
```

To use _DEVICE_, _MANAGED_, and _CUDA\_MANAGED_ you must have the GPU version of MagmaDNN installed. Both managed memory types keep data on the CPU and GPU. By default MagmaDNN performs computations on the GPU, so data must be synced in order to use it on the CPU.

```c++
Tensor<float> y ({100,10}, {UNIFORM,{-1.0f,1.0f}}, MANAGED);

... modify y ...

y.get_memory_manager()->sync(); /* synchronize memory -- sync also takes an optional boolean parameter 'gpu_was_changed' which is defaulted to true */
```

#### Getting/Setting Values
