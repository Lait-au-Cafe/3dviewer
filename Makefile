CXXFLAGS=-std=c++11 -Wall -Wextra -Werror -g
NVCCFLAGS=-Wno-deprecated-gpu-targets
TARGET=3dviewer
GLLIB=-lGLU -lGL -lglfw
CUINC=-I/usr/local/cuda/include
CULIB=-L/usr/local/cuda/lib64/ -lcudart

$(TARGET): source.o viewer.o kernel.o
	nvcc $(NVCCFLAGS) -o $@ $^ $(GLLIB) $(CULIB)

source.o: source.cpp viewer.hpp
	g++ -c $< $(CXXFLAGS) $(CUINC)

kernel.o: kernel.cu kernel.h
	nvcc $(NVCCFLAGS) -c $< $(CUINC) -I/usr/local/cuda/samples/common/inc

viewer.o: viewer.cpp viewer.hpp
	g++ -c $< $(CXXFLAGS) $(CUINC)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.o
