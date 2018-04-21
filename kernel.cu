#include "kernel.h"

__global__ void devStoreVertices(
	float* Vertex,
	const int num_vertex
) {
	const int tx = blockIdx.x*blockDim.x + threadIdx.x;
//	const int ty = blockIdx.y*blockDim.y + threadIdx.y;
	
	const int width = 5;
	uint coord = tx;

	if(coord > num_vertex){
		return;
	}

	float x, y, z;
	x = 0.2 + (tx % width) * 0.1 - 0.5;
	y = 0.2 + (tx / width) * 0.1 - 0.5;
	z = 0;

	coord = 3 * tx;
	Vertex[coord] = x;
	coord = 3 * tx + 1;
	Vertex[coord] = y;
	coord = 3 * tx + 2;
	Vertex[coord] = z;
}

void StoreVertices(
	float* vertex,
	const int num_vertex
){
	// define thread / block size
	dim3 dimBlock(32, 1, 1);
	dim3 dimGrid(num_vertex / dimBlock.x + 1, 1, 1);

	devStoreVertices<<<dimGrid, dimBlock, 0 >>>(vertex, num_vertex);
	return;
}
