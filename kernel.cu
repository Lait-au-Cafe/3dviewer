#include <iostream>
#include <stdio.h>
#include "kernel.h"

__global__ void devStoreVertices(
	float* Vertex,
	const int width,
	const int height,
	const int layers
) {
	const int tx = blockIdx.x*blockDim.x + threadIdx.x;
	const int ty = blockIdx.y*blockDim.y + threadIdx.y;
	const int tz = blockIdx.z*blockDim.z + threadIdx.z;
	
	if(tx >= width || ty >= height || tz >= layers){
		return;
	}

	float vx, vy, vz;
	vx = (float)tx * 0.1 - 0.5;
	vy = (float)ty * 0.1 - 0.5;
	vz = (float)tz * 0.05;

	// rotate a bit
	//vx = 0.95 * vx + 0.31 * vz;
	//vz = -0.31 * vx + 0.95 * vz;

	uint coord = tx + ty * width + tz * width * height;
	uint index;
	index = 3 * coord;
	Vertex[index] = vx;
	index = 3 * coord + 1;
	Vertex[index] = vy;
	index = 3 * coord + 2;
	Vertex[index] = vz;
}

void StoreVertices(
	float* vertex,
	const int width,
	const int height,
	const int layers
){
	// define thread / block size
	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid(
			(width - 1) / dimBlock.x + 1, 
			(height - 1) / dimBlock.y + 1, 
			(layers - 1) / dimBlock.z + 1);

	std::cout
		<< "\n== Configs of StoreVertex ==\n"
		<< "Width : " << width << "\n"
		<< "Height : " << height << "\n"
		<< "Layers : " << layers << "\n"
		<< "Dim of Grid : (" 
			<< dimGrid.x << ", " 
			<< dimGrid.y << ", " 
			<< dimGrid.z << ")\n"
		<< "Dim of Block : (" 
			<< dimBlock.x << ", " 
			<< dimBlock.y << ", " 
			<< dimBlock.z << ")\n"
		<< std::endl;

	devStoreVertices<<<dimGrid, dimBlock, 0 >>>(vertex, width, height, layers);
	return;
}

__global__ void devMLS(
	const float* Input,
	float* Output,
	const int width,
	const int height,
	const int layers,
	const int window,
	const float radius
) {
	const int tx = blockIdx.x*blockDim.x + threadIdx.x;
	const int ty = blockIdx.y*blockDim.y + threadIdx.y;
	const int tz = blockIdx.z*blockDim.z + threadIdx.z;

	if(tx >= width || ty >= height || tz >= layers){
		return;
	}

	const int min_x = max(0, tx - window);
	const int max_x = min(width - 1, tx + window);
	const int min_y = max(0, ty - window);
	const int max_y = min(height - 1, ty + window);
	const int min_z = max(0, tz - window);
	const int max_z = min(layers - 1, tz + window);

	uint coord, index;
	float vx, vy, vz;
	for(int z = min_z; z <= max_z; z++){
	for(int y = min_y; y <= max_y; y++){
	for(int x = min_x; x <= max_x; x++){
		coord = x + y * width + z * width * height;

		index = 3 * coord;
		vx = Input[index];
		index = 3 * coord + 1;
		vy = Input[index];
		index = 3 * coord + 2;
		vz = Input[index];
	}}}

	coord = tx + ty * width + tz * width * height;
	index = 3 * coord;
	vx = Input[index];
	Output[index] = vx;
	index = 3 * coord + 1;
	vy = Input[index];
	Output[index] = vy;
	index = 3 * coord + 2;
	vz = Input[index];
	Output[index] = vz;
}

void MLS(
	const float* const input,
	float* output,
	const int width,
	const int height,
	const int layers,
	const int window,
	const float radius
) {
	// define thread / block size
	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid(
			(width - 1) / dimBlock.x + 1, 
			(height - 1) / dimBlock.y + 1, 
			(layers - 1) / dimBlock.z + 1);

	devMLS<<<dimGrid, dimBlock, 0>>>(input, output, width, height, layers, window, radius);
}
