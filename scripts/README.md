## Scripts
---------------

This folder contains several scripts and utilities for the _development_ of the MagmaDNN source. However, none are essential to developing or using the library.


#### generate_operation.sh

Usage:

``` sh
sh generate_operation.sh "operation_name" y|n
```

This script creates an operation template in the src/include directories. The first parameter is the name of the operation (i.e. "Add" or "Matmul"). The second is y or n, indicating whether to create internal files. Please run the script from the projects root directory.