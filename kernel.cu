#include "kernel.h"

__global__ void devGaussFilter(
	const uchar* Input,
	uchar* Result,
	const int pitch,
	const int width,
	const int height
) {
	const int tx = blockIdx.x*blockDim.x + threadIdx.x;
	const int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int i, j, u, v;
	int coord;

	if (tx >= width || ty >= height) {
		return;
	}

	float sum = 0;
	//float sigma = (float)(GAUSS_WING) / 3;
	float coef;
	uchar data;

	int g_id = 0;
	for (j = -GAUSS_WING; j <= GAUSS_WING; j++) {
		v = min(max(ty + j, 0), height - 1);

		for (i = -GAUSS_WING; i <= GAUSS_WING; i++) {
			u = min(max(tx + i, 0), width - 1);

			coord = u + v * pitch;
			data = Input[coord];

			//coef = expf((i * i + j * j) / (-2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			coef = gaussWindow[g_id];
			g_id++;

			sum += coef * (float)(data);
		}
	}

	coord = tx + ty * pitch;
	Result[coord] = (uchar)min(max((int)sum, 0), 255);
}

void GaussFilter(
	uchar* input,
	uchar* result,
	size_t pitch,
	uint width,
	uint height
){
	// define thread / block size
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	devGaussFilter<<<dimGrid, dimBlock, 0 >>>(input, result, pitch, width, height);
	return;
}
