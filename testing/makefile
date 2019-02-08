# makes the testing files

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))
TARGETS := $(patsubst %.cpp, %.out, $(SRC_FILES))

TESTING_FLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION)
LIB_PATH = ../lib
DEST = ./bin

all: $(TARGETS)

%.out: %.o
	$(CXX) $(TESTING_FLAGS) -o $(DEST)/$(@:.out=) $< $(LIBDIRS) $(LIBS) -L$(LIB_PATH) -lskepsi

%.o: %.cpp
	$(CXX) $(TESTING_FLAGS) -o $@ -c $< $(INC) -I../include

clean:
	rm *.o

# don't remove intermediate .o files
.PRECIOUS: %.o
