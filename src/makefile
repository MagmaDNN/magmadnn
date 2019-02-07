# makes the src files


SRC_FILES = $(wildcard *.cpp)
OBJ_FILES = $(patsubst %.cpp, %.o, $(SRC_FILES))

all: $(OBJ_FILES)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<  -I../include

-include $(OBJ_FILES:.o=.d)