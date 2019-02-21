# makes the testing files

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))
TARGETS := $(patsubst %.cpp, %.out, $(SRC_FILES))

TESTING_FLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION) $(CUDA_MACRO)
RPATH_FLAGS := -Wl,-rpath,$(prefix)/lib
LIB_PATH := $(prefix)/lib
DEST = ./bin

all: $(DEST) $(TARGETS)

$(DEST):
	mkdir -p $@

%.out: %.o
	$(CXX) $(TESTING_FLAGS) $(RPATH_FLAGS) -o $(DEST)/$(@:.out=) $< -L$(LIB_PATH) -lskepsi $(LIBDIRS) $(LIBS)

%.o: %.cpp
	$(CXX) $(TESTING_FLAGS) -o $@ -c $< $(INC) -I../include

clean:
	rm *.o

# don't remove intermediate .o files
.PRECIOUS: %.o
