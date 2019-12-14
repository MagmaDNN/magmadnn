cmake_minimum_required(VERSION 3.9)

project(MagmaDNN LANGUAGES C CXX VERSION 1.0.0 DESCRIPTION "A simple deep learning framework in C++")

# MagmaDNN options
option(MAGMADNN_ENABLE_CUDA "Enable use of CUDA library and compilation of CUDA kernel" OFF)
option(MAGMADNN_ENABLE_MPI "Enable distributed memory routines using MPI" OFF)
option(MAGMADNN_ENABLE_OMP "Enable parallelization using OpenMP library" OFF)
option(MAGMADNN_BUILD_DOC "Generate documentation" OFF)

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
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  CHECK_FORTRAN_FUNCTION_EXISTS("dgemm" BLAS_FOUND)
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
  CHECK_FORTRAN_FUNCTION_EXISTS("dpotrf" LAPACK_FOUND)
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

  enable_language(CUDA)

  add_definitions(-DHAVE_CUDA)
  add_definitions(-DMAGMADNN_HAVE_CUDA)

  set(MAGMADNN_NVCC_ARCHS "60;61;70;75" CACHE STRING "The SM architectures to build code for.")
  
  # Set NVCC arguments
  foreach(ARCH ${REMIFA_NVCC_ARCHS})
    # string(APPEND CUDA_NVCC_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endforeach()

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


