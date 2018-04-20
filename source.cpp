#include "viewer.hpp"
#include "kernel.h"

int main(){
	// Create Viewer
	const int num_vertex = 3;
	Viewer v(num_vertex, "3D Viewer");

	float *vertices;
	size_t pitch;

	v.mapCudaResource((void**)&vertices, &pitch);
	StoreVertices(vertices, num_vertex);
	v.unmapCudaResource();

	// Main Loop
	while(v.update()){}
	return 0;
}
