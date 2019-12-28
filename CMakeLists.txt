cmake_minimum_required(VERSION 3.9)

project(MagmaDNN LANGUAGES C CXX VERSION 1.0.0 DESCRIPTION "A simple deep learning framework in C++")

# include(CheckSymbolExists)
include(CheckFunctionExists)

########################################
# Compiling options

set(default_build_type "RelWithDebInfo")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Using default build type '${default_build_type}' because none was specified")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build (Debug, Release, MinSizeRel, RelWithDebInfo)" FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# MagmaDNN options
option(MAGMADNN_ENABLE_CUDA "Enable use of CUDA library and compilation of CUDA kernel" OFF)
option(MAGMADNN_ENABLE_MPI "Enable distributed memory routines using MPI" OFF)
option(MAGMADNN_ENABLE_OMP "Enable parallelization using OpenMP library" OFF)
option(MAGMADNN_BUILD_DOC "Generate documentation" OFF)
option(MAGMADNN_BUILD_EXAMPLES "Build MagmaDNN examples" ON)
option(MAGMADNN_BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

########################################
# BLAS
set(LBLAS "" CACHE STRING "BLAS library")

# If LBLAS not set, use environement variable
if(LBLAS)
  set(BLAS_LIBRARIES ${LBLAS})
elseif(DEFINED ENV{BLAS_LIB})
  set(BLAS_LIBRARIES $ENV{BLAS_LIB})
endif()

if(DEFINED BLAS_LIBRARIES)
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  check_function_exists("dgemm" BLAS_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT BLAS_FOUND)
    message(ERROR "User supplied BLAS is NOT working")
  endif()
else()
  find_package(BLAS)
endif()

if(BLAS_FOUND)
  set(LIBS ${LIBS} ${BLAS_LIBRARIES})
else()
  message(FATAL_ERROR "BLAS NOT found")
endif (BLAS_FOUND)

########################################
# LAPACK
set(LLAPACK "" CACHE STRING "LAPACK library")
# If LBLAS not set, use environement variable 
if(LLAPACK)
  set(LAPACK_LIBRARIES ${LLAPACK})
elseif(DEFINED ENV{LAPACK_LIB})
  set(LAPACK_LIBRARIES $ENV{LAPACK_LIB})
endif()
if(DEFINED LAPACK_LIBRARIES)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  check_function_exists("dpotrf" LAPACK_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT LAPACK_FOUND)
    message(ERROR "User supplied LAPACK is NOT working")
  endif()

else()
  find_package(LAPACK)
endif()

if (LAPACK_FOUND)
  set(LIBS ${LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
  message(FATAL_ERROR "LAPACK library NOT found")
endif (LAPACK_FOUND)

########################################
# CUDA

if (MAGMADNN_ENABLE_CUDA)

  # find_library(CUDA REQUIRED)

  # MESSAGE( STATUS "CUDA found:         " ${CUDA_FOUND} )
  # MESSAGE( STATUS "CUDA host compiler: " ${CUDA_HOST_COMPILER} )
  # MESSAGE( STATUS "CUDA nvcc compiler: " ${CUDA_NVCC_EXECUTABLE})
  # MESSAGE( STATUS "NVCC FLAGS:         " ${CMAKE_CUDA_FLAGS} )
  # set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  # set(LIBS ${LIBS} ${CUDA_CUBLAS_LIBRARIES})

  enable_language(CUDA)

  # CUDA headers directory
  set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CUDA include dir: ${CUDA_INCLUDE_DIRS}")
  
  # Find cudart library
  find_library(CUDA_RUNTIME_LIBS_DYNAMIC cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_DYNAMIC}" CACHE STRING "Path to a library" FORCE)
  message(STATUS "CUDA runtime library: ${CUDA_RUNTIME_LIBS}")
  set(LIBS ${LIBS} ${CUDA_RUNTIME_LIBS})

  add_definitions(-DHAVE_CUDA)
  add_definitions(-D_HAS_CUDA_)
  add_definitions(-DMAGMADNN_HAVE_CUDA)

  # set(MAGMADNN_NVCC_ARCHS "60;61;70;75" CACHE STRING "The SM architectures to build code for.")
  set(MAGMADNN_NVCC_ARCHS "70" CACHE STRING "The SM architectures to build code for.")
  
  # Set NVCC arguments
  foreach(ARCH ${MAGMADNN_NVCC_ARCHS})
    string(APPEND CUDA_NVCC_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
    # string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endforeach()

  ########################################
  # Find cuBLAS library
  find_library(CUBLAS cublas)

  message(STATUS "cuBLAS: ${CUBLAS}")

  set(LIBS ${LIBS} ${CUBLAS})

  ########################################
  # Find Magma library

  set(MAGMA_DIR "" CACHE STRING "Magam install directory")

  # message(STATUS "ENV{MAGMA_DIR}: $ENV{MAGMA_DIR}")
  
  if(DEFINED ENV{MAGMA_DIR})
    set(MAGMA_DIR $ENV{MAGMA_DIR})
    # message(STATUS "Magma directory: ${MAGMA_DIR}")
  elseif(DEFINED ENV{MAGMADIR})
    set(MAGMA_DIR $ENV{MAGMADIR})
  endif()

  find_library(MAGMA_LIBRARIES magma 
    HINTS ${MAGMA_DIR} /usr/local/magma
    PATH_SUFFIXES lib)

  find_path(MAGMA_INCLUDE_DIRS magma.h 
    HINTS ${MAGMA_DIR} /usr/local/magma
    PATH_SUFFIXES include)

  # find_library(MAGMA)

  if (MAGMA_LIBRARIES)
    set(MAGMA_FOUND TRUE)
  else ()
    SET(MAGMA_FOUND FALSE)
  endif ()
  
  if(MAGMA_FOUND)
    message(STATUS "Magma include dirs: ${MAGMA_INCLUDE_DIRS}")
  else()  
    message(FATAL_ERROR " Magma NOT found")
  endif()

  set(LIBS ${LIBS} ${MAGMA_LIBRARIES})

  ########################################
  # Find cuDNN library

  set(CUDNN_DIR "" CACHE STRING "cuDNN install directory")

  find_library(CUDNN_LIBRARIES cudnn 
    HINTS ${CUDNN_DIR}
    PATH_SUFFIXES lib64)

  find_path(CUDNN_INCLUDE_DIRS cudnn.h 
    HINTS ${CUDNN_DIR}
    PATH_SUFFIXES include)

  if (CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE)
  else ()
    SET(CUDNN_FOUND FALSE)
  endif ()
  
  if(CUDNN_FOUND)
    message(STATUS "cuDNN libraries: ${CUDNN_LIBRARIES}")
    message(STATUS "cuDNN include directories: ${CUDNN_INCLUDE_DIRS}")
  else()  
    message(FATAL_ERROR " cuDNN NOT found")
  endif()

  set(LIBS ${LIBS} ${CUDNN_LIBRARIES})

endif()

########################################
# OpenMP

if (MAGMADNN_ENABLE_OMP)

  find_package(OpenMP REQUIRED)

  if (OPENMP_FOUND)

    add_definitions(-DHAVE_OMP)
    add_definitions(-DMAGMADNN_HAVE_OMP)

  endif()

endif()

########################################

# Load CMake helpers
include(cmake/build_helpers.cmake)
include(cmake/install_helpers.cmake)

########################################
# Subdirectories

add_subdirectory(src)

if(MAGMADNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

magmadnn_install()
