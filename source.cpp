#include "viewer.hpp"
#include "kernel.h"
#include <fstream>

int main(){
	const int width = 400;
	const int height = 200;
	const int layers = 2;
	const int num_vertex = width * height * layers;

	// Create Viewer
	Viewer viewer(num_vertex, "3D Viewer");

	// allocate device memory
	float* d_initial;
	checkCudaErrors(cudaMalloc(&d_initial, num_vertex * 3 * sizeof(float)));

	// initialize buffer
	StoreVertices(d_initial, width, height, layers);

	float *d_vertices; 
	size_t length;

	//===============================
	// Start Mapping >>>
	//===============================
	viewer.mapCudaResource((void**)&d_vertices, &length);
	std::cout << "Mapped CUDA Buffer:"
		<< length << "Byte" << std::endl;
	if(length != num_vertex * 3 * sizeof(float)){
		std::cerr 
			<< "The size of the memory mapped is "
			<< "different from that requested. \n"
			<< "Requested : " 
				<< num_vertex * 3 * sizeof(float) << "Bytes/n"
			<< "Mapped : " << length << "Bytes/n"
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	//	kernel execution
	//StoreVertices(d_vertices, width, height, layers);
	int window = 2;
	float radius = 100;
	MLS(d_initial, d_vertices, width, height, layers, window, radius);

	// collect the result
	float *vertices = (float*)malloc(num_vertex * 3 * sizeof(float));
	checkCudaErrors(cudaMemcpy(
		(void*)vertices, 
		(void*)d_vertices, 
		length, 
		cudaMemcpyDeviceToHost));

	viewer.unmapCudaResource();
	//===============================
	// <<< End Mapping
	//===============================

//	std::ofstream logfile("data_raw.csv");
//	logfile << "x,y,z,r,g,b" << std::endl;
//	// print vertex coordinates
//	int index = 0;
//	for(int j = 0; j < layers; j++){
//		std::cout << "\nLayer " << j+1 << std::endl;
//		for(int i = 0; i < width * height; i++){
//			std::cout
//				<< i+1 << " : ("
//				<< vertices[3 * index] << ", "
//				<< vertices[3 * index + 1] << ", "
//				<< vertices[3 * index + 2] << ")"
//				<< std::endl;
//			logfile 
//				<< vertices[3 * index] << ","
//				<< vertices[3 * index + 1] << ","
//				<< vertices[3 * index + 2] << ","
//				<< (j == 0 ? 1.0 : 0.0) << ","
//				<< (j == 0 ? 0.0 : 1.0) << ","
//				<< 0.0 << std::endl;
//			index++;
//		}
//	}

	// Main Loop
	while(viewer.update()){}
	return 0;
}
