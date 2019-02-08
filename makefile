# This makefile can build and install the skepsi library

include make.inc

# compilers
CXX ?= g++
NVCC ?= nvcc


CUDADIR ?= /usr/local/cuda
MAGMADIR ?= /usr/local/magma

# install location
prefix ?= /usr/local/skepsi

# headers needed for library compilation
INC = -I./include 
LIBDIRS =
LIBS = 

# do we have cuda installed?
ifneq ($(shell which nvcc),)
CUDA_MACRO = -D_HAS_CUDA_
INC += -I$(CUDADIR)/include
LIBDIRS += -L$(CUDADIR)/lib64
LIBS += -lcudart
endif


# individual flags for compilation
OPTIMIZATION_LEVEL ?= -O3
WARNINGS ?= -Wall
FPIC ?= -fPIC
CXX_VERSION ?= -std=c++11

# the entire flags for compilation
CXXFLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION) $(CUDA_MACRO) $(FPIC) -MMD
NVCCFLAGS := $(OPTIMIZATION_LEVEL) -Xcompiler "$(CXXFLAGS)"


# make these available to child makefiles
export CXX
export NVCC
export INC
export LIBDIRS
export LIBS
export CUDADIR
export MAGMADIR
export OPTIMIZATION_LEVEL
export WARNINGS
export CXX_VERSION
export CXXFLAGS
export NVCCFLAGS

o_ext ?= o
TARGET_DIRS = src

all: $(TARGET_DIRS)

$(TARGET_DIRS):
	@echo "==== Building Sources ===="
	$(MAKE) -C $@
	@echo


OBJ_FILES := $(wildcard $(TARGET_DIRS)/*.o)


# make the libs

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
	$(MAKE) -C $(TESTING_DIR)


install: $(TARGET_DIRS) lib
	@echo "==== installing libs ===="
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	cp -r ./include/* $(prefix)/include
	cp $(libstatic) $(prefix)/lib
	cp $(libshared) $(prefix)/lib
	@echo


clean:
	rm $(wildcard $(TARGET_DIRS)/*.o)


.PHONY: $(TARGET_DIRS) testing


