#include "kernel.h"

__global__ void devStoreVertices(
	float* Vertex,
	const int num
) {
	const int tx = blockIdx.x*blockDim.x + threadIdx.x;
	const int ty = blockIdx.y*blockDim.y + threadIdx.y;
	
	const int width = 3;
	uint coord;

	float x, y, z;
	x = 0.2 + (tx % width) * 0.1;
	y = 0.2 + (ty / width) * 0.1;
	z = 0;

	coord = 3 * tx;
	Vertex[coord] = x;
	coord = 3 * tx + 1;
	Vertex[coord] = y;
	coord = 3 * tx + 2;
	Vertex[coord] = z;
}

void StoreVertices(
	float* input,
	const int num
){
	// define thread / block size
	dim3 dimBlock(32, 1, 1);
	dim3 dimGrid(num / dimBlock.x, 1, 1);

	devStoreVertices<<<dimGrid, dimBlock, 0 >>>(input, num);
	return;
}
