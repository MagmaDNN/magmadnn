## Tutorial 03: Compute Graph
-----------------------------
Operations in MagmaDNN are done in compute graphs. A compute graph represents a calculation as an abstract syntax tree. For example consider the below tree. This represents the expression `WX + B`.

```
         add
       /     \
    matmul    B
    /    \
  W        X
```

It is important to note that compute graphs are _not_ computed immediately in MagmaDNN. The graph is first constructed and then must be explicitly evaluated to compute the output.

All nodes in the compute graph inherit from the class `::magmadnn::op::Operation<T>`. Likewise all operations are in the `::magmadnn::op` namespace. Each operation has a helper functions that will create and setup the object for you. 

The most basic type of operation is the `op::Variable<T>`. Variables simply wrap tensors and represent them as nodes in the compute graph. They can be created with `op::var("variable name", tensor_ptr);` or you can let the variable instantiate the tensor for you `op::var("variable name", {tensor, shape}, {tensor, {filler}}, memory_type);`. 

For example, the above compute graph of `WX+B`:

```c++
/* first, initialize the variables */
op::Operation<float> *w = op::var("W", {10,5}, {UNIFORM, {0.0f, 1.0f}}, HOST);

/* the auto keyword is helpful when initializing operations */
auto x = op::var("X", {5,6}, {UNIFORM, {0.0f, 1.0f}}, HOST);
auto b = op::var("B", {10,6}, {UNIFORM, {0.0f, 1.0f}}, HOST);


/* this constructs the compute graph */
auto out = op::add(op::matmul(w,x), b);
```

Note, however, that this does not evaluate the compute graph (i.e. compute the resulting value). To do so we must call the operations `eval` function.

```c++
Tensor<float> *out_tensor = out->eval();
```

Now out_tensor will store the output of the compute graph. 

It is important to clear the compute graph from memory, which can be done simply by deleting the head node.

```c++
delete out; /* clears the compute graph from memory */
```

To see more supported operations and their respective documentation please build the docs as shown in [tutorial 00](docs/tutorials/00_installing.md).

