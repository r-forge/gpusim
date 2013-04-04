
#include "utils.h"


/*******************************************************************************************
** GPU KERNELS *****************************************************************************
********************************************************************************************/

#define NUM_BLOCKS  4
#define NUM_THREADS 512
// -> Total 4 * 512 Threads --> each thread computes n/(4*512) elements

// distance function
__global__ void dist(double *out, double *a, double *b, int n) {
	__shared__ double sum[NUM_THREADS];
	sum[threadIdx.x] = 0.0;

	int elementsPerThread = ceil((double)n/(NUM_THREADS*NUM_BLOCKS));
	for (int i=blockIdx.x * NUM_THREADS * elementsPerThread + threadIdx.x; i < n && i < ((blockIdx.x+1) * NUM_THREADS * elementsPerThread); i+=NUM_THREADS) {
		sum[threadIdx.x] += (a[i] - b[i]) * (a[i] - b[i]);
		
	}
	__syncthreads();
	
	// reduction
	for (int i=NUM_THREADS >> 1; i>=1; i >>= 1) {
		if (threadIdx.x < i) sum[threadIdx.x] += sum[threadIdx.x+i];
		__syncthreads();
	}
	
	if (threadIdx.x == 0) out[blockIdx.x] = sum[0];
}




// C oder C++ Compiler
#ifdef __cplusplus
extern "C" {
#endif

//alle Funktionen, die mit EXPORT gekennzeichnet sind, koennen in R ueber .C aufgerufen werden


void EXPORT gpuDist(double *res, double *a, double *b, int* n) {
	double *d_a;
	double *d_b;
	double *d_out;
	
	cudaMalloc((void**)&d_a,sizeof(double)* (*n));
	cudaMalloc((void**)&d_b,sizeof(double)* (*n));
	cudaMalloc((void**)&d_out,sizeof(double)* NUM_BLOCKS);
	double *out = (double*)malloc(sizeof(double)*NUM_BLOCKS);

	cudaMemcpy(d_a, a, *n * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, *n * sizeof(double),cudaMemcpyHostToDevice);
	
	
	dim3 blockSize = dim3(NUM_THREADS);	
	dim3 blockCount = dim3(NUM_BLOCKS);
	
	// TODO: test whether both works, n= multiple of blocksize*blockcount or not
	
	dist<<<blockCount, blockSize>>>(d_out, d_a, d_b, *n);
	
	// sum all workgroups
	
	cudaMemcpy(out, d_out, NUM_BLOCKS * sizeof(double),cudaMemcpyDeviceToHost);
	double sum=0.0;
	for (int i=0; i<NUM_BLOCKS; ++i)  sum += out[i];
	
	free(out);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	
	res[0] = sum;
}


void EXPORT cpuDistMatrix(double *res, double *a, int *nx, int*ny, int*nz) {
	//double *out = (double*)malloc(sizeof(double)*(*nz)*(*nz));

	for (int i=0;i<*nz; ++i) {
		for (int j=i; j<*nz; ++j) {
			res[i*(*nz)+j] = 0.0;
			if (i == j) continue;
			for (int k=0; k<(*nx * (*ny)); ++k) {
				res[i*(*nz)+j] += (a[i*(*nx)*(*ny) + k] -  a[j*(*nx)*(*ny) + k]) * (a[i*(*nx)*(*ny) + k] -  a[j*(*nx)*(*ny) + k]); 
			}
			res[j*(*nz)+i] = res[i*(*nz)+j];
		}
	}
}






#ifdef __cplusplus
}
#endif
