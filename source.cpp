#include "viewer.hpp"
#include "kernel.h"

int main(){
	const int num_vertex_per_layer = 50;
	const int width = 5;
	const int layers = 2;
	const int num_vertex = num_vertex_per_layer * layers;

	// Create Viewer
	Viewer v(num_vertex, "3D Viewer");

	float *d_vertices;
	size_t length;

	float *vertices = (float*)malloc(num_vertex * 3 * sizeof(float));

	v.mapCudaResource((void**)&d_vertices, &length);
	std::cout << "Allocated CUDA Buffer:"
		<< length << "Byte" << std::endl;
	if(length != num_vertex * 3 * sizeof(float)){
		std::cerr 
			<< "The size of the memory allocated is "
			<< "different from that requested. \n"
			<< "Requested : " << num_vertex * 3 * sizeof(float) << "Bytes/n"
			<< "Allocated : " << length << "Bytes/n"
			<< std::endl;
		exit(EXIT_FAILURE);
	}

//	checkCudaErrors(cudaMemcpy((void*)d_vertices, (void*)vertices, length, cudaMemcpyHostToDevice));
//	for(int i = 0; i < (int)length >> 2; i++){
//		std::cout << i << ":" << vertices[i] << std::endl;
//	}

	//	kernel execution
	StoreVertices(d_vertices, width, num_vertex_per_layer, layers);

	// collect the result
	checkCudaErrors(cudaMemcpy(
		(void*)vertices, 
		(void*)d_vertices, 
		length, 
		cudaMemcpyDeviceToHost));
	for(int i = 0; i < (int)(length / sizeof(float)); i+=3){
		std::cout 
			<< i/3+1 << " : ("
			<< vertices[i] << ", "
			<< vertices[i+1] << ", "
			<< vertices[i+2] << ")" << std::endl;
	}

	v.unmapCudaResource();

	// Main Loop
	while(v.update()){}
	return 0;
}
