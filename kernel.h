#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GAUSS_WING 3
#define GAUSS_SIZE ((2 * GAUSS_WING + 1) * (2 * GAUSS_WING + 1))

typedef unsigned char uchar;

//=========================================================
// Global Buffers
//=========================================================
__constant__ float gaussWindow[GAUSS_SIZE];

//=========================================================
// Device Functions
//=========================================================
void StoreVertices(float*, const int, const int);
void MLS(float*, const int, const int, const int, const int, const float);
