CXXFLAGS=-std=c++11 -Wall -Wextra -Werror -g
TARGET=3dviewer
GLLIBS=-lGLU -lGL -lglfw
CUINC=-I/usr/local/cuda/include
CULIBS=-L/usr/local/cuda/lib64/ -lcudart

$(TARGET): source.o viewer.o
	g++ -o $@ $^ $(GLLIBS) $(CULIBS)

source.o: source.cpp viewer.hpp
	g++ -c $< $(CXXFLAGS) $(CUINC)

viewer.o: viewer.cpp viewer.hpp
	g++ -c $< $(CXXFLAGS) $(CUINC)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.o
