# makes the example files

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))
TARGETS := $(patsubst %.cpp, %.out, $(SRC_FILES))

TESTING_FLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION) $(CUDA_MACRO)
RPATH_FLAGS := -Wl,-rpath,$(prefix)/lib
LIB_PATH := $(prefix)/lib
DEST = ./bin

USE_HDF5 ?= 0
ifeq ($(USE_HDF5),1)
TESTING_FLAGS += -DUSE_HDF5
HDF5DIR ?= /usr/local/hdf5
TESTING_FLAGS += -I$(HDF5DIR)/include
LIBDIRS += -L$(HDF5DIR)/lib
LIBS += -lhdf5 -lhdf5_cpp -lhdf5_hl
endif

ifeq ($(HAS_CUDA),1)
TESTING_FLAGS += -DUSE_GPU
endif

ifeq ($(DEBUG),1)
TESTING_FLAGS += -g -DDEBUG
endif

all: $(DEST) $(TARGETS)

$(DEST):
	mkdir -p $@

%.out: %.o
	$(CXX) $(TESTING_FLAGS) $(RPATH_FLAGS) -o $(DEST)/$(@:.out=) $< -L$(LIB_PATH) -lmagmadnn $(LIBDIRS) $(LIBS)

%.o: %.cpp
	$(CXX) $(TESTING_FLAGS) -o $@ -c $< $(INC) -I../include

clean:
	rm *.o

