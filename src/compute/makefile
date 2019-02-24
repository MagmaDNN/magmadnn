# makes the src files


SRC_FILES = $(wildcard *.cpp */*.cpp)
OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC_FILES))

ifeq ($(USE_CUDA),1)
CU_FILES = $(wildcard *.cu */*.cu)
CU_OBJ_FILES = $(patsubst %.cu,%.o,$(CU_FILES))
endif

SUB_DIRS =

all: $(SUB_DIRS) $(CU_OBJ_FILES) $(OBJ_FILES)

$(SUB_DIRS):
	$(MAKE) -C $@

$(CU_OBJ_FILES): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< $(INC) -I../../include


$(OBJ_FILES): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<  $(INC) -I../../include 

.PHONY: $(SUB_DIRS)

-include $(OBJ_FILES:.o=.d)
