## Tutorial 01: Hello World
---------------------------
Make sure you have MagmaDNN installed and set up properly. If not please consult the previous tutorial [00_installing](https://github.com/MagmaDNN/magmadnn/tree/master/docs/tutorials/00_installing.md). 

This tutorial will walk you writing a simple program that initializes MagmaDNN, prints `hello world`, and finalizes MagmaDNN. It also focuses on how to compile your MagmaDNN programs.

#### Hello World
First, the code:
```c++
#include <cstdio>
#include "magmadnn.h"

using namespace magmadnn;

int main() {
    magmadnn_init();

    std::printf("Hello World!\n");

    magmadnn_finalize();
}
```

Lets break this down into parts. First we include the magmadnn header files using

```c++
#include "magmadnn.h"
```

This allows your program to see MagmaDNN's functions/classes and use them in your code. 

Next we declare we're using the MagmaDNN namespace. 

```c++
using namespace magmadnn;
```

This allows to not explicitly include the namespace in every magmadnn call (i.e. `magmadnn::magmadnn_init()`). It is not required, but is a helpful shorthand in your code writing.

Inside the `main` function, we call 2 magmadnn functions:

```c++
magmadnn_init();
... code ...
magmadnn_finalize();
```

In order to use MagmaDNN, you must begin your code with `magmadnn_init` and end it with `magmadnn_finalize`. These calls allow MagmaDNN to initialize Magma, gather other information about the runtime environment, and initialize some settings.


#### Compiling
The MagmaDNN compilation is similar to that of any other c++ code. You must include the MagmaDNN headers using `-I` and link to the lib files using `-l` and `-L`.

On the CPU:
```sh
g++ -O3 -o hello_world -I/path/to/blas/include -I/path/to/magmadnn/include hello_world.cpp -L/path/to/blas/lib -L/path/to/magmadnn/lib -lopenblas -lmagmadnn
```

On the GPU:
```sh
g++ -O3 -DUSE_GPU -o hello_world -I/path/to/blas/include -I/path/to/magma/include -I/path/to/magmadnn/include hello_world.cpp -L/path/to/magma/lib -L/path/to/blas/lib -L/path/to/magmadnn/lib -lopenblas -lcudart -lcudnn -lmagma -lmagmadnn
```

or a more general makefile

```makefile
CXX = g++
INC = -I/path/to/blas/include -I/path/to/magma/include -I/path/to/magmadnn/include
FLAGS = -O3 $(INC) -DUSE_GPU
LIBS = -L/path/to/magma/lib -L/path/to/blas/lib -L/path/to/magmadnn/lib -lopenblas -lcudart -lcudnn -lmagma -lmagmadnn
TARGETS = hello_world

all: $(TARGETS)

hello_world: hello_world.cpp
    $(CXX) $(FLAGS) -o $@ $< $(LIBS)

clean:
    rm $(TARGETS)
```