#### Adding New Operations 
---------------------------
The skeleton of the operation superclass can be found at [include/compute/operation.h](https://github.com/Dando18/skepsi/blob/master/include/compute/operation.h). There are several things that must be implemented in order to have a working operation. Below the process is layed out for the example operation `foo`.

1. Create a new folder in `include/compute` and `src/compute` named `foo`.
2. Inside the `include/compute/foo` folder create `foo_op.h` and in `src/compute/foo` create `foo_op.cpp`. 
3. Define the `foo_op` class, which extends `operation`, in the header file. Note that the class must be in the namespace `skepsi::op`. The method `foo` should also be defined here, which returns a new operation class `foo`.
4. Implement `foo_op` in `foo_op.cpp`. The new operation must implement the constructor, eval, and to_string methods. `eval` should return the evaluated tensor up to this point in the tree. (Note, you are allowed to create helper files within the folder `foo_op/`, but their methods must live within the namespace `skepsi::internal`). Implement the function `foo` in `foo_op.cpp`. `foo` must work with and be compiled for `int`, `float`, and `double`. `foo` is also expected to work for all memory types.
5. Add `#include "foo_op/foo_op.h"` to `include/compute/tensor_operations.h`. This allows the rest of the library to see the new operation.
6. _Optional:_ Add a tester file to the `testing/` folder.

See [include/compute/add/](https://github.com/Dando18/skepsi/tree/master/include/compute/add) and [src/compute/add/](https://github.com/Dando18/skepsi/tree/master/src/compute/add) for an example.

Operators _should_ implement copy and no-copy options, determining whether to return a newly allocated tensor or write over one of the parameters.