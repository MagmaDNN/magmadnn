# This makefile can build and install the magmadnn library

# incluse any user-defined compile options (pre-sets)
include make.inc

# compilers for c++ and cu
CXX ?= g++
NVCC ?= nvcc

# locations of cuda and magma installations
CUDADIR ?= /usr/local/cuda
MAGMADIR ?= /usr/local/magma
BLASDIR ?= /usr/local/openblas
BLASINC ?= $(BLASDIR)/include
BLASLIB_PATH ?= $(BLASDIR)/lib
BLASLIB ?= openblas

# where to install magmadnn (make must have sudo access if prefix is root privileged)
prefix ?= /usr/local/magmadnn

# headers needed for library compilation
INC := -I./include -I$(BLASINC)

# libs to link with
LIBDIRS := -L$(BLASLIB_PATH)
LIBS = -l$(BLASLIB)

# use nvcc to determine if we should compile for gpu or not
USE_CUDA = 0
GPU_TARGET ?= Kepler
ifneq ($(shell which nvcc),)
include make.device
CUDA_MACRO = -D_HAS_CUDA_
INC += -I$(CUDADIR)/include -I$(MAGMADIR)/include
LIBDIRS += -L$(CUDADIR)/lib64 -L$(MAGMADIR)/lib
LIBS += -lcudart -lcudnn -lmagma
USE_CUDA=1
endif


# individual flags for compilation
OPTIMIZATION_LEVEL ?= -O3
WARNINGS ?= -Wall
FPIC ?= -fPIC
CXX_VERSION ?= -std=c++11
DEBUG ?= 0
PROFILE_FLAGS ?= 

# this flag dictates whether to use openmp or not for some CPU code
# set it to false by default
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
OPENMP_FLAGS = -fopenmp -D_USE_OPENMP_
endif

# set optimization to Og for debugging
ifeq ($(DEBUG),1)
OPTIMIZATION_LEVEL = -O0
endif

# the entire flags for compilation
CXXFLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION) $(CUDA_MACRO) $(FPIC) -MMD $(OPENMP_FLAGS) $(PROFILE_FLAGS)
NVCCFLAGS := $(CXX_VERSION) $(OPTIMIZATION_LEVEL) -Xcompiler "$(CXXFLAGS)" $(NV_SM) $(NV_COMP)
LD_FLAGS := $(LIBDIRS) $(LIBS)

# include -g for debugging
ifeq ($(DEBUG),1)
CXXFLAGS += -g -DDEBUG
NVCCFLAGS += -g -DDEBUG
endif

# make these available to child makefiles
export CXX
export NVCC
export INC
export LD_FLAGS
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
TARGET_DIRS ?= src

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

# collect all the object files from the source directories (deepest: src/compute/*/*.cpp)
OBJ_FILES = $(wildcard $(TARGET_DIRS)/*.$(o_ext) $(TARGET_DIRS)/*/*.$(o_ext) $(TARGET_DIRS)/*/*/*.$(o_ext))
# collect dependency files (.d)
DEP_FILES = $(wildcard $(TARGET_DIRS)/*.d $(TARGET_DIRS)/*/*.d $(TARGET_DIRS)/*/*/*.d)


# MAKE THE LIB FILES

LIB_DIR ?= lib

# archiver and flags
ARCH ?= ar
ARCH_FLAGS ?= cr
RANLIB ?= ranlib
RANLIB_FLAGS ?= 

# set EXT to .dylib and FLAG to -dynamiclib for MAC OS
LIBSHARED_EXT ?= .so
LIBSHARED_FLAG ?= -shared

libstatic := $(LIB_DIR)/libmagmadnn.a
libshared := $(LIB_DIR)/libmagmadnn$(LIBSHARED_EXT)

lib: $(TARGET_DIRS) static shared
static: $(libstatic)
shared: $(libshared)

$(libstatic): 
	@echo "==== building static lib ===="
	mkdir -p $(LIB_DIR)
	$(ARCH) $(ARCH_FLAGS) $@ $(OBJ_FILES)
	$(RANLIB) $(RANLIB_FLAGS) $@
	@echo 


# MacOS specific install information
ostype = ${shell echo $${OSTYPE}}
ifneq ($(findstring darwin, ${ostype}),)
    $(libshared): LIBSHARED_FLAG += -install_name $(prefix)/lib/$(notdir $(libshared))
endif

$(libshared):
	@echo "==== building shared lib ===="
	mkdir -p $(LIB_DIR)
	$(CXX) $(LIBSHARED_FLAG) $(FPIC) -o $@ $(OBJ_FILES) -L./lib $(LD_FLAGS)
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
install: lib
	@echo "==== installing libs ===="
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	cp -r ./include/* $(prefix)/include
	cp $(libstatic) $(prefix)/lib
	cp $(libshared) $(prefix)/lib
	@echo


# build the docs files and the refman.pdf
DOCS_DIR ?= docs
docs:
ifneq ($(shell which doxygen),)
	doxygen doxygen.config
	$(MAKE) -C $(DOCS_DIR)/latex
endif


# TODO: change to call clean on subdirectory makefiles
clean:
	rm $(OBJ_FILES) $(DEP_FILES)


.PHONY: $(TARGET_DIRS) $(libstatic) $(libshared) $(TESTING_DIR) $(EXAMPLE_DIR) $(DOCS_DIR)


