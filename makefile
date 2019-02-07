# This makefile can build and install the danielnn library

include make.inc

# compilers
CXX ?= g++
NVCC ?= nvcc

CUDADIR ?= /usr/local/cuda
MAGMADIR ?= /usr/local/magma

# install location
prefix ?= /usr/local/danielnn

# headers needed for library compilation
INC = -I./include
LIBDIRS = 
LIBS = 

# individual flags for compilation
OPTIMIZATION_LEVEL ?= -O3
WARNINGS ?= -Wall
FPIC ?= -fPIC
CXX_VERSION = -std=c++11

# the entire flags for compilation
CXXFLAGS = $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION) $(FPIC) -MMD
NVCCFLAGS = $(OPTIMIZATION_LEVEL) -Xcompiler "$(FPIC) $(WARNINGS)"

# make these available to child makefiles
export CXX
export NVCC
export OPTIMIZATION_LEVEL
export WARNINGS
export CXX_VERSION
export CXXFLAGS
export NVCCFLAGS

o_ext ?= o
TARGET_DIRS = src

all: $(TARGET_DIRS)

$(TARGET_DIRS):
	$(MAKE) -C $@


OBJ_FILES := $(wildcard $(TARGET_DIRS)/*.o)


# make the libs

# archiver and flags
ARCH ?= ar
ARCH_FLAGS ?= cr
RANLIB ?= ranlib
RANLIB_FLAGS ?= -no_warning_for_no_symbols


libstatic ?= lib/libdanielnn.a
libshared ?= lib/libdanielnn.so

lib: static shared
static: $(libstatic)
shared: $(libshared)


$(libstatic): 
	@echo "==== building static lib ===="
	$(ARCH) $(ARCH_FLAGS) $@ $(OBJ_FILES)
	$(RANLIB) $(RANLIB_FLAGS) $@
	@echo 

$(libshared):
	@echo "==== building shared lib ===="
	$(CXX) -shared -o $@ $(OBJ_FILES) -L./lib 
	@echo 


# make the testers
TESTING_DIR ?= testing
testing:
	$(MAKE) -C $(TESTING_DIR)

install: $(TARGET_DIRS) lib
	@echo "==== installing libs ===="
	cp -r ./include $(prefix)/include
	cp $(libstatic) $(prefix)/lib
	cp $(libshared) $(prefix)/lib
	@echo

clean:
	rm $(wildcard $(TARGET_DIRS)/*.o)


.PHONY: $(TARGET_DIRS) testing


