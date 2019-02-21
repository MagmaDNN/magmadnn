# This makefile can build and install the skepsi library

# incluse any user-defined compile options (pre-sets)
include make.inc

# compilers for c++ and cu
CXX ?= g++
NVCC ?= nvcc

# locations of cuda and magma installations
CUDADIR ?= /usr/local/cuda
MAGMADIR ?= /usr/local/magma

# where to install skepsi (make must have sudo access if prefix is root privileged)
prefix ?= /usr/local/skepsi

# headers needed for library compilation
INC = -I./include 

# libs to link with
LIBDIRS =
LIBS = 

# use nvcc to determine if we should compile for gpu or not
USE_CUDA = 0
GPU_TARGET ?= Kepler
ifneq ($(shell which nvcc),)
include make.device
CUDA_MACRO = -D_HAS_CUDA_
INC += -I$(CUDADIR)/include
LIBDIRS += -L$(CUDADIR)/lib64
LIBS += -lcudart
USE_CUDA=1
endif


# individual flags for compilation
OPTIMIZATION_LEVEL ?= -O3
WARNINGS ?= -Wall
FPIC ?= -fPIC
CXX_VERSION ?= -std=c++11

# the entire flags for compilation
CXXFLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION) $(CUDA_MACRO) $(FPIC) -MMD
NVCCFLAGS := $(CXX_VERSION) $(OPTIMIZATION_LEVEL) -Xcompiler "$(CXXFLAGS)" $(NV_SM) $(NV_COMP)


# make these available to child makefiles
export CXX
export NVCC
export INC
export LIBDIRS
export LIBS
export prefix
export CUDADIR
export MAGMADIR
export OPTIMIZATION_LEVEL
export WARNINGS
export CXX_VERSION
export CUDA_MACRO
export USE_CUDA
export CXXFLAGS
export NVCCFLAGS


# default extension for object files
o_ext ?= o

# where are the source files (files that need to be compiled) found
TARGET_DIRS = src

all: $(TARGET_DIRS)

# step into source directories and use their makefiles
$(TARGET_DIRS):
	@echo "==== Building Sources ===="
ifeq ($(USE_CUDA),1)
	@echo "CUDA installation found. Building GPU/CPU."
else
	@echo "(X) CUDA installation not found. Building CPU-only."
endif
	$(MAKE) -C $@
	@echo

# collect all the object files from the source directories
OBJ_FILES = $(wildcard $(TARGET_DIRS)/*.o $(TARGET_DIRS)/*/*.o)


# MAKE THE LIB FILES

# archiver and flags
ARCH ?= ar
ARCH_FLAGS ?= cr
RANLIB ?= ranlib
RANLIB_FLAGS ?= 

# set EXT to .dylib and FLAG to -dynamiclib for MAC OS
LIBSHARED_EXT ?= .so
LIBSHARED_FLAG ?= -shared

libstatic := lib/libskepsi.a
libshared := lib/libskepsi$(LIBSHARED_EXT)

lib: static shared
static: $(libstatic)
shared: $(libshared)

$(libstatic): 
	@echo "==== building static lib ===="
	mkdir -p lib
	$(ARCH) $(ARCH_FLAGS) $@ $(OBJ_FILES)
	$(RANLIB) $(RANLIB_FLAGS) $@
	@echo 

$(libshared):
	@echo "==== building shared lib ===="
	mkdir -p lib
	$(CXX) $(LIBSHARED_FLAG) $(FPIC) -o $@ $(OBJ_FILES) -L./lib 
	@echo 


# make the testers
TESTING_DIR ?= testing
testing:
	@echo "==== building testing sources ===="
	# step into the testing directories and call their makefiles
	$(MAKE) -C $(TESTING_DIR)
	@echo

# make the examples
EXAMPLE_DIR ?= examples
examples:
	@echo "==== building examples ===="
	# step into example directory and use its makefile
	$(MAKE) -C $(EXAMPLE_DIR)
	@echo


# build the library first, then link the lib together.
# install copies the newly made libs into prefix
install: $(TARGET_DIRS) lib
	@echo "==== installing libs ===="
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	cp -r ./include/* $(prefix)/include
	cp $(libstatic) $(prefix)/lib
	cp $(libshared) $(prefix)/lib
	@echo

# TODO: change to call clean on subdirectory makefiles
clean:
	rm $(wildcard $(TARGET_DIRS)/*.o $(TARGET_DIRS)/*/*.o)


.PHONY: $(TARGET_DIRS) testing examples


