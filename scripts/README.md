## Scripts
---------------

This folder contains several scripts and utilities for the _development_ of the MagmaDNN source. None are essential to developing or using the library.


#### generate_operation.sh

Usage:

``` sh
sh scripts/generate_operation.sh "operation_name" y|n
```

This script creates an operation template in the src/include directories. The first parameter is the name of the operation (i.e. "Add" or "Matmul"). The second is y or n, indicating whether to create internal files. Please run the script from the projects root directory.


#### build_and_test.sh

Usage:

``` sh
sh scripts/build_and_test.sh ...flags...
```

This script can clean, build, install, and run the testers for the MagmaDNN library. It is mainly intended to assist in the development of the library. Available flags:

| Flag      	| Shortened Version 	|  Takes Argument 	| Definition                                                             	|
|-----------	|-------------------	|-----------------	|------------------------------------------------------------------------	|
|  --help   	| -h                	|                 	| prints the help options and usage information                          	|
| --threads 	| -j                	|        ✓        	|  specifies the number of threads to compile MagmaDNN with.             	|
| --debug   	| -d                	|                 	|  if set, MagmaDNN will be compile with the   DEBUG flag set.           	|
| --output  	| -o                	|        ✓        	| output file location                                                   	|
| --clean   	| -c                	|                 	|  if set, then `make clean` will be run before building and installing. 	|
| --verbose 	| -v                	|                 	| verbose mode                                                           	|
| --no-test 	| N/A               	|                 	|  if set, then the testers will not be built or run.                    	|
| --dev         | N/A                   |                   | same as '--clean --debug --verbose'                                       |
