# makes the src files
CUDA_SRC_FILES = $(wildcard cuda/*.cpp)
MKLDNN_SRC_FILES = $(wildcard onednn/*.cpp)
SRC_FILES = $(filter-out $(wildcard cuda/*.cpp onednn/*.cpp),$(wildcard *.cpp */*.cpp))

ifeq ($(USE_MKLDNN),1)
SRC_FILES += $(MKLDNN_SRC_FILES)
endif

ifeq ($(USE_CUDA),1)
CU_FILES = $(wildcard *.cu */*.cu)
CU_OBJ_FILES = $(patsubst %.cu,%.o,$(CU_FILES))
SRC_FILES += $(CUDA_SRC_FILES)
endif

OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC_FILES))

SUB_DIRS =


all: $(SUB_DIRS) $(CU_OBJ_FILES) $(OBJ_FILES)

TEST:
	$(MKLDNN_SRC_FILES)

$(SUB_DIRS):
	$(MAKE) -C $@

$(CU_OBJ_FILES): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< $(INC) -I../../include


$(OBJ_FILES): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<  $(INC) -I../../include 

.PHONY: $(SUB_DIRS)

-include $(OBJ_FILES:.o=.d)
