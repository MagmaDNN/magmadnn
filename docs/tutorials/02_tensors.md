## Tutorial 02: Tensors
-----------------------

Tensors are fundamental to machine learning operations. They store multi-dimensional data and are the basis for all the math operations in MagmaDNN.


MagmaDNN defines a `Tensor` class which stores and keeps track of tensors for us. To create one, simply use:

```c++
/* create a  10x5  tensor */
Tensor<float> example_tensor ({10,5});
```

The above creates a 10x5 tensor called `example_tensor`.