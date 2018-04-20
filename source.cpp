#include "viewer.hpp"
#include "kernel.h"

int main(){
	// Create Viewer
	const int num_vertex = 3;
	Viewer v(num_vertex, "3D Viewer");

	float *d_vertices;
	size_t length;

	float *vertices = (float*)malloc(num_vertex * 3 * sizeof(float));

	v.mapCudaResource((void**)&d_vertices, &length);
	std::cout << length << std::endl;
//	checkCudaErrors(cudaMemcpy((void*)d_vertices, (void*)vertices, length, cudaMemcpyHostToDevice));
//	for(int i = 0; i < (int)length >> 2; i++){
//		std::cout << i << ":" << vertices[i] << std::endl;
//	}
	StoreVertices(d_vertices, num_vertex);
	checkCudaErrors(cudaMemcpy((void*)vertices, (void*)d_vertices, length, cudaMemcpyDeviceToHost));
	for(int i = 0; i < (int)(length / sizeof(float)); i++){
		std::cout << i << ":" << vertices[i] << std::endl;
	}
	v.unmapCudaResource();

	// Main Loop
	while(v.update()){}
	return 0;
}
