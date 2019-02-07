# makes the testing files

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SRC_FILES))
TARGETS := $(patsubst %.cpp, %.out, $(SRC_FILES))

TESTING_FLAGS := $(OPTIMIZATION_LEVEL) $(WARNINGS) $(CXX_VERSION)
LIB_PATH = ../lib

all: $(TARGETS)

%.out: %.o
	$(CXX) $(TESTING_FLAGS) -o $(@:.out=) $< $(LIBDIRS) $(LIBS) -L$(LIB_PATH) -ldanielnn

%.o: %.cpp
	$(CXX) $(TESTING_FLAGS) -o $@ -c $< $(INC) -I../include

clean:
	rm *.o