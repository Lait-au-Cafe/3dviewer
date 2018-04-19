CXXFLAGS=-std=c++11 -Wall -Wextra -Werror -g
GLLIBS=-lGLU -lGL -lglfw
TARGET=3dviewer

$(TARGET): source.o viewer.o
	g++ -o $@ $^ $(GLLIBS)

viewer.o: viewer.cpp viewer.hpp
	g++ -c $< $(CXXFLAGS)

source.o: source.cpp viewer.hpp
	g++ -c $< $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.o
