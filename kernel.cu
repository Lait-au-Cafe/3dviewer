#include <iostream>
#include <stdio.h>
#include "kernel.h"
#include <helper_math.h>

#define PI 3.14159265358979323846
inline __device__ float radians(float x){ return x * PI / 180; }

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

	uint seed = tx + ty * width + tz * width * height;
	seed = seed ^ (seed << 13);
	seed = seed ^ (seed >> 17);
	seed = seed ^ (seed << 15);
	float val = (float)(seed % 10000) / 10000;

	float vx, vy, vz;
	vx = (float)tx * 0.05 + (tz * 0.01);
	vy = (float)ty * 0.05;
	//vz = (float)tz;
	vz = val;

	// scale
	float scale = 0.25;
	vx *= scale;
	vy *= scale;
	vz *= scale;

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

inline __device__ float mls_weight(float distance, float radius){ 
	return (distance > radius) ? 0 : powf(1-powf(distance / radius, 2), 4); 
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
	float3 ctr_pos, v_pos, rel_pos, accum_pos = make_float3(0.0f, 0.0f, 0.0f);
	float w, sum = 0;

	coord = tx + ty * width + tz * width * height;
	index = 3 * coord;
	ctr_pos.x = Input[index];
	index = 3 * coord + 1;
	ctr_pos.y = Input[index];
	index = 3 * coord + 2;
	ctr_pos.z = Input[index];

	for(int z = min_z; z <= max_z; z++){
	for(int y = min_y; y <= max_y; y++){
	for(int x = min_x; x <= max_x; x++){
		coord = x + y * width + z * width * height;

		index = 3 * coord;
		v_pos.x = Input[index];
		index = 3 * coord + 1;
		v_pos.y = Input[index];
		index = 3 * coord + 2;
		v_pos.z = Input[index];

		rel_pos = v_pos - ctr_pos;
		w = mls_weight(length(rel_pos), radius);
		accum_pos += v_pos * w;
		sum += w;
	}}}

	float3 avrg_pos = accum_pos / sum;
	float3 norm = make_float3(0.0, 0.0, 1.0);
	float3 res_pos = ctr_pos + dot(norm, avrg_pos - ctr_pos) * norm;
	//res_pos = ctr_pos;
	coord = tx + ty * width + tz * width * height;
	index = 3 * coord;
	Output[index] = res_pos.x;
	index = 3 * coord + 1;
	Output[index] = res_pos.y;
	index = 3 * coord + 2;
	Output[index] = res_pos.z;
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
