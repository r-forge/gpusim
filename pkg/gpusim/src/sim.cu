/**
* sim.cu C and CUDA Interface for gpusim R package
* Author: Marius Appel - marius.appel@uni-muenster.de
**/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <curand.h>
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include <fstream>
#include <cuda.h>
#include <string>


#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define PI 3.1415926535897932384626433832795 
enum covfunc {EXP=0, SPH=1, GAU=2}; 


/* Helper function for writing a raster to a csv file for testing purposes */
void writeCSVMatrix(const char *filename, float* matrix, int numRows, int numCols) {
	using namespace std;
	
	fstream file;
	file.open(filename, ios::out);
	if (file.fail()) return;

	for (int i=0; i<numRows; ++i) {
		for (int j = 0; j<numCols; ++j) {
			file << matrix[i * numCols + j] << " ";
		}
		file << "\n";
	}
	file.close();
}


#ifdef __cplusplus
extern "C" {
#endif
// Covariance functions
void EXPORT covExp(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		out[i] = ((h[i] == 0.0)? (*nugget + *sill) : (*sill*exp(-h[i]/(*range))));
	}
}
void EXPORT covGau(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		out[i] = ((h[i] == 0.0)? (*nugget + *sill) : (*sill*exp(- (h[i]*h[i]) / ((*range)*(*range))  )));
	}
}
void EXPORT covSph(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		if (h[i] == 0.0) 
			out[i] =  (*nugget + *sill);	
		else if(h[i] <= *range) 
			out[i] = *sill * (1.0 - (((3*h[i]) / (2*(*range))) - ((h[i] * h[i] * h[i]) / (2 * (*range) * (*range) * (*range))) ));	
		else out[i] = 0.0; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
	}
}

#ifdef __cplusplus
}
#endif



/*******************************************************************************************
** GPU KERNELS *****************************************************************************
********************************************************************************************/

// Covariance functions
__device__ float covExpKernel(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/range)));
}
__device__ float covGauKernel(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist2 = (ax-bx)*(ax-bx)+(ay-by)*(ay-by);
	return ((dist2 == 0.0f)? (nugget + sill) : (sill*expf(-dist2/(range*range))));
}
__device__ float covSphKernel(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0f - (((3.0f*dist) / (2.0f*range)) - ((dist * dist * dist) / (2.0f * range * range * range)) ));	
	return 0.0f; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}





// Converts real float array into cufftComplex array
__global__ void realToComplexKernel(cufftComplex *c, float* r, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		c[i].x = r[i];
		c[i].y = 0.0f;
	}
}


// Gets relevant real parts of 2nx x 2ny grid and devides it by div
__global__ void ReDiv(float *out, cufftComplex *c, float div,int nx, int ny) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//if (col < nx && row < ny) out[col*ny+row] = c[col*2*ny+row].x / div; 
	if (col < nx && row < ny) out[row*nx+col] = c[row*2*nx+col].x / div; 
}


// Covariance sampling of a regular grid
__global__ void sampleCovExpKernel(cufftComplex *trickgrid, cufftComplex *grid, cufftComplex* cov, float xc, float yc, float sill, float range, float nugget, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		cov[row*n+col].x = covExpKernel(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
		cov[row*n+col].y = 0;	
		if (col == n/2-1 && row == m/2-1) {
			trickgrid[row*n+col].x = 1.0f;
			trickgrid[row*n+col].y = 0.0f;
		}
		else {
			trickgrid[row*n+col].x = 0.0f;
			trickgrid[row*n+col].y = 0.0f;
		}
	}
}
// Covariance sampling of a regular grid
__global__ void sampleCovGauKernel(cufftComplex *trickgrid, cufftComplex *grid, cufftComplex* cov, float xc, float yc, float sill, float range, float nugget, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		cov[row*n+col].x = covGauKernel(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
		cov[row*n+col].y = 0;	
		if (col == n/2-1 && row == m/2-1) {
			trickgrid[row*n+col].x = 1.0f;
			trickgrid[row*n+col].y = 0.0f;
		}
		else {
			trickgrid[row*n+col].x = 0.0f;
			trickgrid[row*n+col].y = 0.0f;
		}
	}
}


// Covariance sampling of a regular grid
__global__ void sampleCovSphKernel(cufftComplex *trickgrid, cufftComplex *grid, cufftComplex* cov, float xc, float yc, float sill, float range, float nugget, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		cov[row*n+col].x = covSphKernel(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
		cov[row*n+col].y = 0;	
		if (col == n/2-1 && row == m/2-1) {
			trickgrid[row*n+col].x = 1.0f;
			trickgrid[row*n+col].y = 0.0f;
		}
		else {
			trickgrid[row*n+col].x = 0.0f;
			trickgrid[row*n+col].y = 0.0f;
		}
	}
}




// Multiplies a grid per cell with m*n
__global__ void multKernel(cufftComplex *fftgrid, int n, int m) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	fftgrid[i].x = fftgrid[i].x*n*m;
	fftgrid[i].y = fftgrid[i].y*n*m;
}

// Devides spectral grid elementwise by fftgrid
__global__ void divideSpectrumKernel(cufftComplex *spectrum, cufftComplex *fftgrid) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float a = spectrum[i].x;
	float b = spectrum[i].y;
	float c = fftgrid[i].x;
	float d = fftgrid[i].y;
	spectrum[i].x = (a*c+b*d)/(c*c+d*d);
	spectrum[i].y = (b*c-a*d)/(c*c+d*d);
}

// Element-wise sqrt from spectral grid
__global__ void sqrtKernel(cufftComplex *spectrum) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float re = spectrum[i].x;
	float im = spectrum[i].y;
	float sill = 0;
	float d = sqrt(re*re+im*im);
	float dsqrt = sqrt(d);
	if(re>0)
		sill = atan(im/re);
	if(re<0 && im>=0)
		sill = atan(im/re)+PI;
	if(re<0 && im<0)
		sill = atan(im/re)-PI;
	if(re==0 && im>0)
		sill = PI/2;
	if(re==0 && im<0)
		sill = -PI/2;
	spectrum[i].x = dsqrt*cos(sill/2);
	spectrum[i].y = dsqrt*sin(sill/2);
}

// Element-wise multiplication of two complex arrays
__global__ void elementProduct(cufftComplex *c, cufftComplex *a, cufftComplex *b, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		c[i].x = a[i].x * b[i].x - a[i].y * b[i].y;
		c[i].y = a[i].x * b[i].y + a[i].y * b[i].x;
	}
}


/// Kriging prediction at a regular grid with given samples for conditioning
#ifndef BLOCK_SIZE_KRIGE1
#define BLOCK_SIZE_KRIGE1 256
#endif

__global__ void krigingExpKernel(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  float range, float sill, float nugget, int numSrc, int nx, int ny)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0f;
	float yr_x, yr_y;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][2];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny){
		yr_x = ((bx*BLOCK_SIZE_KRIGE1 + tx) % nx) * dx + xmin; // grid coords
		yr_y = (ny - 1 -(int)((bx*BLOCK_SIZE_KRIGE1 + tx)/nx)) * dy + ymin; // grid coords 
	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					sum += covExpKernel(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i];                         
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}


__global__ void krigingGauKernel(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  float range, float sill, float nugget, int numSrc, int nx, int ny)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0f;
	float yr_x, yr_y;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][2];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny){
		yr_x = ((bx*BLOCK_SIZE_KRIGE1 + tx) % nx) * dx + xmin; // grid coords
		yr_y = (ny - 1 - (int)((bx*BLOCK_SIZE_KRIGE1 + tx)/nx)) * dy + ymin; // grid coords
	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					sum += covGauKernel(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 	                         
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}

__global__ void krigingSphKernel(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  float range, float sill, float nugget, int numSrc, int nx, int ny)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0f;
	float yr_x, yr_y;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][2];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny){
		yr_x = ((bx*BLOCK_SIZE_KRIGE1 + tx) % nx) * dx + xmin; // grid coords
		yr_y = (ny - 1 - (int)((bx*BLOCK_SIZE_KRIGE1 + tx)/nx)) * dy + ymin; // grid coords
	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					sum += covSphKernel(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 	                         
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}


// Adds interpolated residuals element-wise to the unconditional grid. Residual grid will be overwritten in-place
__global__ void addResSim(float *res, float *uncond, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id];
}



// Overlay of samples and regular grid. For each sample, the underlying grid cell in subpixel accuracy is returned, so that a following bilinear 
// interpolation is possible
__global__ void overlay(float2 *out, float2 *xy, float grid_minx, float grid_dx, float grid_maxy, float grid_dy, int numPoints) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPoints) {
		out[i].x = (xy[i].x - grid_minx)/grid_dx;
		out[i].y = (grid_maxy - grid_dy - xy[i].y)/grid_dy;
	}
}


// Calculates residuals of samples and an unconditional realization. Uses bilinear interpolation based on the sample's position in grid
__global__ void residuals(float* res, float *srcdata, float *uncond_grid, float2 *indices, int nx, int ny, int numPoints) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numPoints) {
		
		// Bilinear interpolation
		float x = indices[id].x; 
		float y = indices[id].y;
		int row = floor(y); // y index of upper neighbour pixel
		int col = floor(x); // x index of lower neighbour pixel
		x = (float)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (float)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour

		// Special case: last column / row
		if (col > nx-1) {
			x = 0.0f;col = nx-1;
		}
		else if (col < 0) {
			x = 0.0f;col=0;
		}
		if (row > nx-1) {
			y = 0.0f;row = nx-y;
		}	
		else if (row < 0) {
			y = 0.0f;row=0;
		}
		res[id] = srcdata[id] - ((1-y) * ((1-x) * uncond_grid[row * nx + col] + x * uncond_grid[row * nx + col + 1]) + 
								  y * ((1-x) * uncond_grid[(row+1) * nx + col] + x * uncond_grid[(row+1) * nx + col + 1]));
	}		
	if (id == 0) {
		res[numPoints] = 0.0f; // Needed as Lagrange factor for GEMV with inverse covariance matrix of samples (needed for Kriging)
	}
}


#ifdef __cplusplus
extern "C" {
#endif

	void EXPORT init(int *result) {
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)  {
			printf("cudaSetDevice returned error code %d\n", cudaStatus);
			*result = 1;
		}
		*result = 0;
	}


#ifdef __cplusplus
}
#endif




/*******************************************************************************************
** UNCONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/

// global variables for unconditional simulation. These data are needed in the preprocessing as well as in generating realizations
struct uncond_state {
	cufftComplex *d_cov; // d_cov is the result of the preprocessing ans is needed for each realozation
	int nx,ny,n,m;
	float xmin,xmax,ymin,ymax,dx,dy;
	int blockSize,numBlocks;
	dim3 blockSize2, numBlocks2;
	cufftHandle plan1;
	dim3 blockSize2d;
	dim3 blockCount2d;
	dim3 blockSize1d;
	dim3 blockCount1d;
} uncond_global;

#ifdef __cplusplus
extern "C" {
#endif

void EXPORT unconditionalSimInit(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, float *p_sill, float *p_range, float *p_nugget, int *p_covmodel, int *ret_code) {
	*ret_code = 1;
	cudaError_t cudaStatus;
	
	uncond_global.nx= *p_nx; // Number of cols
	uncond_global.ny= *p_ny; // Number of rows
	uncond_global.n= 2*uncond_global.nx; // Number of cols
	uncond_global.m= 2*uncond_global.ny; // Number of rows
	uncond_global.dx = (*p_xmax - *p_xmin) / uncond_global.nx;
	uncond_global.dy = (*p_ymax - *p_ymin) / uncond_global.ny;
	
	// 1d cuda grid
	uncond_global.blockSize1d = dim3(256);
	uncond_global.blockCount1d = dim3(uncond_global.n*uncond_global.m / uncond_global.blockSize1d.x);
	if (uncond_global.n * uncond_global.m % uncond_global.blockSize1d.x  != 0) ++uncond_global.blockCount1d.x;
	
	// 2d cuda grid
	uncond_global.blockSize2d = dim3(16,16);
	uncond_global.blockCount2d = dim3(uncond_global.n / uncond_global.blockSize2d.x, uncond_global.m / uncond_global.blockSize2d.y);
	if (uncond_global.n % uncond_global.blockSize2d.x != 0) ++uncond_global.blockCount2d.x;
	if (uncond_global.m % uncond_global.blockSize2d.y != 0) ++uncond_global.blockCount2d.y;

	
	//cufftPlan2d(&uncond_global.plan1, uncond_global.n, uncond_global.m, CUFFT_C2C); 
	cufftPlan2d(&uncond_global.plan1, uncond_global.m, uncond_global.n, CUFFT_C2C); 

	
	// build grid (ROW MAJOR)
	cufftComplex *h_grid_c = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global.m*uncond_global.n);
	for (int i=0; i<uncond_global.n; ++i) { // i =  col index
		for (int j=0; j<uncond_global.m; ++j) { // j = row index 
			h_grid_c[j*uncond_global.n+i].x = *p_xmin + i * uncond_global.dx; 
			h_grid_c[j*uncond_global.n+i].y = *p_ymin + j * uncond_global.dy;  
		}
	}

	
	float xc = (uncond_global.dx*uncond_global.n)/2;
	float yc = (uncond_global.dy*uncond_global.m)/2;
	float sill = *p_sill;
	float range = *p_range;
	float nugget = *p_nugget;
	cufftComplex *d_grid;
	
	// Array for grid
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(cufftComplex)*uncond_global.n*uncond_global.m);
	// Array for cov grid
	cudaStatus = cudaMalloc((void**)&uncond_global.d_cov,sizeof(cufftComplex)*uncond_global.n*uncond_global.m);

	// Sample covariance and generate "trick" grid
	cufftComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftComplex)*uncond_global.n*uncond_global.m);
	
	// copy grid to GPU
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, uncond_global.n*uncond_global.m*sizeof(cufftComplex),cudaMemcpyHostToDevice);
	
	switch(*p_covmodel) {
	case EXP:
		sampleCovExpKernel<<<uncond_global.blockCount2d, uncond_global.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global.d_cov, xc, yc, sill, range,nugget,uncond_global.n,uncond_global.m);
		break;
	case GAU:
		sampleCovGauKernel<<<uncond_global.blockCount2d, uncond_global.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global.d_cov, xc, yc, sill, range,nugget,uncond_global.n,uncond_global.m);
		break;
	case SPH:
		sampleCovSphKernel<<<uncond_global.blockCount2d, uncond_global.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global.d_cov, xc, yc, sill, range,nugget,uncond_global.n,uncond_global.m);
		break;
	}
	free(h_grid_c);
	cudaFree(d_grid);


	// Execute 2d FFT of covariance grid in order to get the spectral representation 
	cufftExecC2C(uncond_global.plan1, uncond_global.d_cov, uncond_global.d_cov, CUFFT_FORWARD); // in place fft forward
	cufftExecC2C(uncond_global.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forward

	// Copy to host and check for negative real parts (NOT WORKING YET)
	/*cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global.n*uncond_global.m);
	cudaStatus = cudaMemcpy(h_cov,uncond_global.d_cov,sizeof(cufftComplex)*uncond_global.n*uncond_global.m,cudaMemcpyDeviceToHost);
	for (int i=0; i<uncond_global.n*uncond_global.m; ++i) {
		if (h_cov[i].x < 0.0) {
			*ret_code = 2; 
			free(h_cov);
			cudaFree(d_trick_grid_c);
			cudaFree(uncond_global.d_cov);
			return;
		}	
	}
	free(h_cov);*/
	
	// Multiply fft of "trick" grid with n*m
	multKernel<<<uncond_global.blockCount1d, uncond_global.blockSize1d>>>(d_trick_grid_c, uncond_global.n, uncond_global.m);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel!\n", cudaStatus);	

	// Devide spectral covariance grid by "trick" grid
	divideSpectrumKernel<<<uncond_global.blockCount1d, uncond_global.blockSize1d>>>(uncond_global.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	// Compute sqrt of cov grid
	sqrtKernel<<<uncond_global.blockCount1d, uncond_global.blockSize1d>>>(uncond_global.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel\n", cudaStatus);

	*ret_code = 0;
}

// Generates unconditional realizations
// p_out = output array of size nx*ny*k * sizeof(float)
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT unconditionalSimRealizations(float *p_out,  int *p_k, int *ret_code)
{
	*ret_code = 1;
	cudaError_t cudaStatus;

	int k = *p_k;

	float *d_rand; // device random numbers
	curandGenerator_t curandGen;
	cufftComplex *d_fftrand;
	cufftComplex* d_amp;
	float* d_out;

	curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGen,(unsigned long long)(time(NULL)));	

	for(int l = 0; l<k; ++l) {
		
		if(l==0){			
			cudaStatus = cudaMalloc((void**)&d_rand,sizeof(float)*uncond_global.m*uncond_global.n); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
			// Generate Random Numbers
		curandGenerateNormal(curandGen,d_rand,uncond_global.m*uncond_global.n,0.0,1.0);
		
		if(l==0) {
			cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftComplex) * uncond_global.n * uncond_global.m); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
		// Convert real random numbers to complex numbers
		realToComplexKernel<<< uncond_global.blockCount1d, uncond_global.blockSize1d>>>(d_fftrand, d_rand, uncond_global.n*uncond_global.m);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel!\n", cudaStatus);	

		// Compute 2D FFT of random numbers
		cufftExecC2C(uncond_global.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward
		

		if(l==0) cudaMalloc((void**)&d_amp,sizeof(cufftComplex)*uncond_global.n*uncond_global.m);
		elementProduct<<<uncond_global.blockCount1d, uncond_global.blockSize1d>>>(d_amp, uncond_global.d_cov, d_fftrand, uncond_global.m*uncond_global.n);  
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct!\n", cudaStatus);
		
		cufftExecC2C(uncond_global.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation
		
		if(l==0) cudaMalloc((void**)&d_out,sizeof(float)*uncond_global.nx*uncond_global.ny);
		
		dim3 blockSize2dhalf  = dim3(16,16);
		dim3 blockCount2dhalf = dim3(uncond_global.nx/blockSize2dhalf.x,uncond_global.ny/blockSize2dhalf.y);
		if (uncond_global.nx % blockSize2dhalf.x != 0) ++blockCount2dhalf.x;
		if (uncond_global.ny % blockSize2dhalf.y != 0) ++blockCount2dhalf.y;
		ReDiv<<<blockCount2dhalf, blockSize2dhalf>>>(d_out, d_amp, std::sqrt((float)(uncond_global.n*uncond_global.m)), uncond_global.nx, uncond_global.ny);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess) {
			printf("cudaThreadSynchronize returned error code %d after launching ReDiv!\n", cudaStatus);
		}

		cudaMemcpy((p_out + l*(uncond_global.nx*uncond_global.ny)),d_out,sizeof(float)*uncond_global.nx*uncond_global.ny,cudaMemcpyDeviceToHost);					
	}

	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_out);
	curandDestroyGenerator(curandGen);
	*ret_code = 0;
}


void EXPORT unconditionalSimRelease(int *ret_code) {
	*ret_code = 1;
	cudaFree(uncond_global.d_cov);
	cufftDestroy(uncond_global.plan1);
	*ret_code = 0;
}


#ifdef __cplusplus
} // extern C
#endif






/*******************************************************************************************
** CONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/


// global variables for conditional simulation that are needed both, for initialization as well as for generating realizations
struct cond_state {
	cufftComplex *d_cov; 
	int nx,ny,n,m;
	float xmin,xmax,ymin,ymax,dx,dy;
	float range, sill, nugget;
	int blockSize,numBlocks;
	dim3 blockSize2, numBlocks2;
	cufftHandle plan1;
	dim3 blockSize2d;
	dim3 blockCount2d;
	dim3 blockSize1d;
	dim3 blockCount1d;
	dim3 blockSizeSamples;
	dim3 blockCountSamples;
	dim3 blockSizeSamplesPlus1;
	dim3 blockCountSamplesPlus1;
	// Variables for conditioning
	int numSrc; // Number of sample observation
	float2 *d_samplexy; // coordinates of samples
	float2 *d_sampleindices; // Corresponding grid indices in subpixel accuracy
	float *d_sampledata; // data values of samples
	float *d_covinv; // inverse covariance matrix of samples
	int covmodel;

} cond_global;



#ifdef __cplusplus
extern "C" {
#endif


void EXPORT conditionalSimInit(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, float *p_sill, float *p_range, float *p_nugget, float *p_srcXY, float *p_srcData, int *p_numSrc, float *p_cov_inv, int *p_covmodel, int *ret_code) {
	*ret_code = 1;
	cudaError_t cudaStatus;
	cublasInit();

	cond_global.nx= *p_nx; // Number of cols
	cond_global.ny= *p_ny; // Number of rows
	cond_global.n= 2*cond_global.nx; // Number of cols
	cond_global.m= 2*cond_global.ny; // Number of rows
	cond_global.dx = (*p_xmax - *p_xmin) / cond_global.nx;
	cond_global.dy = (*p_ymax - *p_ymin) / cond_global.ny;
	cond_global.numSrc = *p_numSrc;
	cond_global.xmin = *p_xmin;
	cond_global.xmax = *p_xmax;
	cond_global.ymin = *p_ymin;
	cond_global.ymax = *p_ymax;
	cond_global.range = *p_range;
	cond_global.sill = *p_sill;
	cond_global.nugget = *p_nugget;
	cond_global.covmodel = *p_covmodel;

	// 1d cuda grid
	cond_global.blockSize1d = dim3(256);
	cond_global.blockCount1d = dim3(cond_global.n*cond_global.m / cond_global.blockSize1d.x);
	if (cond_global.n * cond_global.m % cond_global.blockSize1d.x  != 0) ++cond_global.blockCount1d.x;
	
	// 2d cuda grid
	cond_global.blockSize2d = dim3(16,16);
	cond_global.blockCount2d = dim3(cond_global.n / cond_global.blockSize2d.x, cond_global.m / cond_global.blockSize2d.y);
	if (cond_global.n % cond_global.blockSize2d.x != 0) ++cond_global.blockCount2d.x;
	if (cond_global.m % cond_global.blockSize2d.y != 0) ++cond_global.blockCount2d.y;

	// 1d cuda grid for samples
	cond_global.blockSizeSamples = dim3(256);
	cond_global.blockCountSamples = dim3(cond_global.numSrc / cond_global.blockSizeSamples.x);
	if (cond_global.numSrc % cond_global.blockSizeSamples.x !=0) ++cond_global.blockCountSamples.x;

	// Setup fft
	//cufftPlan2d(&cond_global.plan1, cond_global.n, cond_global.m, CUFFT_C2C); // n und m vertauscht weil col major grid
	cufftPlan2d(&cond_global.plan1, cond_global.m, cond_global.n, CUFFT_C2C); // n und m vertauscht weil col major grid

	
	// Build grid (ROW MAJOR)
	cufftComplex *h_grid_c = (cufftComplex*)malloc(sizeof(cufftComplex)*cond_global.m*cond_global.n);
	for (int i=0; i<cond_global.n; ++i) { // i = col index
		for (int j=0; j<cond_global.m; ++j) { // j = row index
			h_grid_c[j*cond_global.n+i].x = *p_xmin + i * cond_global.dx; 
			h_grid_c[j*cond_global.n+i].y = *p_ymin + j * cond_global.dy; 
		}
	}

	
	float xc = (cond_global.dx*cond_global.n)/2;
	float yc = (cond_global.dy*cond_global.m)/2;
	cufftComplex *d_grid;
	
	// Allocate grid and cov arrays on GPU
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(cufftComplex)*cond_global.n*cond_global.m);
	cudaStatus = cudaMalloc((void**)&cond_global.d_cov,sizeof(cufftComplex)*cond_global.n*cond_global.m);

	// Sample covariance and generate "trick" grid
	cufftComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftComplex)*cond_global.n*cond_global.m);
	
	// Copy grid to gpu
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, cond_global.n*cond_global.m*sizeof(cufftComplex),cudaMemcpyHostToDevice);
	
	switch(cond_global.covmodel) {
	case EXP:
		sampleCovExpKernel<<<cond_global.blockCount2d, cond_global.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global.d_cov, xc, yc, cond_global.sill, cond_global.range, cond_global.nugget, cond_global.n,cond_global.m);
		break;
	case GAU:
		sampleCovGauKernel<<<cond_global.blockCount2d, cond_global.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global.d_cov, xc, yc, cond_global.sill, cond_global.range, cond_global.nugget, cond_global.n,cond_global.m);
		break;
	case SPH:
		sampleCovSphKernel<<<cond_global.blockCount2d, cond_global.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global.d_cov, xc, yc, cond_global.sill, cond_global.range, cond_global.nugget, cond_global.n,cond_global.m);
		break;
	}

	free(h_grid_c);
	cudaFree(d_grid);


	// Compute spectral representation of cov and "trick" grid
	cufftExecC2C(cond_global.plan1, cond_global.d_cov, cond_global.d_cov, CUFFT_FORWARD); // in place fft forward
	cufftExecC2C(cond_global.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forward

	// Copy to host and check for negative real parts // NOT WORKING YET
	/*cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*cond_global.n*cond_global.m);
	cudaStatus = cudaMemcpy(h_cov,cond_global.d_cov,sizeof(cufftComplex)*cond_global.n*cond_global.m,cudaMemcpyDeviceToHost);
	for (int i=0; i<cond_global.n*cond_global.m; ++i) {
		if (h_cov[i].x < 0.0) {
			*ret_code = 2; 
			free(h_cov);
			cudaFree(d_trick_grid_c);
			cudaFree(cond_global.d_cov);
			return;
		}	
	}
	free(h_cov);*/

	// Multiplication of fft(trick_grid) with n*m	
	multKernel<<<cond_global.blockCount1d, cond_global.blockSize1d>>>(d_trick_grid_c, cond_global.n, cond_global.m);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel!\n", cudaStatus);	

	// Devide spectral cov grid by fft of "trick" grid
	divideSpectrumKernel<<<cond_global.blockCount1d, cond_global.blockSize1d>>>(cond_global.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	// Compute sqrt of spectral cov grid
	sqrtKernel<<<cond_global.blockCount1d, cond_global.blockSize1d>>>(cond_global.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel\n", cudaStatus);

	// Copy samples to gpu
	cudaStatus = cudaMalloc((void**)&cond_global.d_samplexy,sizeof(float2)* cond_global.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global.d_sampleindices,sizeof(float2)*cond_global.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global.d_sampledata,sizeof(float)*cond_global.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMemcpy(cond_global.d_samplexy,p_srcXY,sizeof(float2)* cond_global.numSrc,cudaMemcpyHostToDevice);
	cudaMemcpy(cond_global.d_sampledata,p_srcData,sizeof(float)*cond_global.numSrc,cudaMemcpyHostToDevice);
		

	// Overlay samples to grid and save resulting subpixel grind indices
	overlay<<<cond_global.blockCountSamples,cond_global.blockSizeSamples>>>(cond_global.d_sampleindices,cond_global.d_samplexy,*p_xmin,cond_global.dx,*p_ymax,cond_global.dy, cond_global.numSrc);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay!\n", cudaStatus);	
	// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access			

	/* TEST OUTPUT ON HOST */
	/*float2 *h_indices = (float2*)malloc(sizeof(float2)*cond_global.numSrc);
	cudaMemcpy(h_indices,cond_global.d_sampleindices,sizeof(float2)*cond_global.numSrc,cudaMemcpyDeviceToHost);
	for (int i=0;i<cond_global.numSrc;++i) {
		printf("(%.2f,%.2f) -> (%.2f,%.2f)\n",p_srcXY[2*i],p_srcXY[2*i+1],h_indices[i].x, h_indices[i].y);
	}
	free(h_indices);*/


	// copy inverse sample cov matrix to gpu
	cudaMalloc((void**)&cond_global.d_covinv,sizeof(float) * (cond_global.numSrc + 1) * (cond_global.numSrc + 1));
	cudaMemcpy(cond_global.d_covinv,p_cov_inv,sizeof(float) * (cond_global.numSrc + 1) * (cond_global.numSrc + 1),cudaMemcpyHostToDevice);

	*ret_code = 0;
}




// Generates conditional realizations
// p_out = output array of size nx*ny*k * sizeof(float)
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT conditionalSimRealizations(float *p_out, int *p_k, int *ret_code)
{
	*ret_code = 1;
	cudaError_t cudaStatus;
	
	int k = *p_k;
	
	
	float *d_rand; // Device Random Numbers
	curandGenerator_t curandGen;
	cufftComplex *d_fftrand;
	cufftComplex* d_amp;	
	float* d_uncond;    // grid of one unconditional realization
	float *d_residuals; // residuals of samples and underlying unconditional realization
	float *d_y; // result vector from solving the kriging equation system
	float *d_respred; // interpolated residuals
	
	curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGen,(unsigned long long)(time(NULL)));	

	for(int l = 0; l<k; ++l) {
			
		if(l==0){	
			
			cudaStatus = cudaMalloc((void**)&d_rand,sizeof(float)*cond_global.m*cond_global.n); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
	
		curandGenerateNormal(curandGen,d_rand,cond_global.m*cond_global.n,0.0,1.0);

		if(l==k-1) {
			curandDestroyGenerator(curandGen);
		}
	
		if(l==0) {
			cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftComplex) * cond_global.n * cond_global.m); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}


		realToComplexKernel<<< cond_global.blockCount1d, cond_global.blockSize1d>>>(d_fftrand, d_rand, cond_global.n*cond_global.m);

		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel!\n", cudaStatus);	

		cufftExecC2C(cond_global.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward
		cudaStatus = cudaThreadSynchronize();

		if(l==0) cudaMalloc((void**)&d_amp,sizeof(cufftComplex)*cond_global.n*cond_global.m);
		
		elementProduct<<<cond_global.blockCount1d, cond_global.blockSize1d>>>(d_amp, cond_global.d_cov, d_fftrand, cond_global.m*cond_global.n);
    
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct!\n", cudaStatus);

		cufftExecC2C(cond_global.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation

		 
		if(l==0) cudaMalloc((void**)&d_uncond,sizeof(float)*cond_global.nx*cond_global.ny);

		dim3 blockSize2dhalf  = dim3(16,16);
		dim3 blockCount2dhalf = dim3(cond_global.nx/blockSize2dhalf.x,cond_global.ny/blockSize2dhalf.y);
		if (cond_global.nx % blockSize2dhalf.x != 0) ++blockCount2dhalf.x;
		if (cond_global.ny % blockSize2dhalf.y != 0) ++blockCount2dhalf.y;
		ReDiv<<<blockCount2dhalf, blockSize2dhalf>>>(d_uncond, d_amp, std::sqrt((float)(cond_global.n*cond_global.m)), cond_global.nx, cond_global.ny);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess) {
			printf("cudaThreadSynchronize returned error code %d after launching ReDiv!\n", cudaStatus);
		}
		// d_uncond is now a unconditional realization 
		// Compute residuals between samples and d_uncond
		if (l==0) cudaMalloc((void**)&d_residuals,sizeof(float)* (cond_global.numSrc + 1));
		residuals<<<cond_global.blockCountSamples,cond_global.blockSizeSamples>>>(d_residuals,cond_global.d_sampledata,d_uncond,cond_global.d_sampleindices,cond_global.nx,cond_global.ny,cond_global.numSrc);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
		
		// Residuals are now stored in d_residuals 
		// Interpolate residuals	
		
		// Compute matrix vector product y = d_covinv * d_residuals on gpu using cublas	
		if (l==0) cudaMalloc((void**)&d_y,sizeof(float)*(cond_global.numSrc + 1));
		
		cublasSgemv('n',cond_global.numSrc + 1,cond_global.numSrc + 1, 1.0, cond_global.d_covinv,cond_global.numSrc + 1,d_residuals,1,0.0,d_y,1);
	
		// Kriging prediction
		if (l==0)  cudaMalloc((void**)&d_respred, sizeof(float) * cond_global.nx * cond_global.ny);
		dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
		dim3 blockCntKrige = dim3((cond_global.nx*cond_global.ny) / blockSizeKrige.x);
		if ((cond_global.nx*cond_global.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;


		switch(cond_global.covmodel) {
		case EXP:
			krigingExpKernel<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global.d_samplexy,cond_global.xmin,cond_global.dx,cond_global.ymin,cond_global.dy,d_y,cond_global.range,cond_global.sill,cond_global.nugget,cond_global.numSrc,cond_global.nx,cond_global.ny);
			break;
		case GAU:
			krigingGauKernel<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global.d_samplexy,cond_global.xmin,cond_global.dx,cond_global.ymin,cond_global.dy,d_y,cond_global.range,cond_global.sill,cond_global.nugget,cond_global.numSrc,cond_global.nx,cond_global.ny);
			break;
		case SPH:
			krigingSphKernel<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global.d_samplexy,cond_global.xmin,cond_global.dx,cond_global.ymin,cond_global.dy,d_y,cond_global.range,cond_global.sill,cond_global.nugget,cond_global.numSrc,cond_global.nx,cond_global.ny);
			break;
		}


		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel!\n", cudaStatus);
		
		// Add result to unconditional realization
		dim3 blockSizeCond = dim3(256);
		dim3 blockCntCond = dim3(cond_global.nx*cond_global.ny/ blockSizeCond.x);
		if (cond_global.nx*cond_global.ny % blockSizeCond.x != 0) ++blockSizeCond.x;
		addResSim<<<blockCntCond,blockSizeCond>>>(d_respred, d_uncond, cond_global.nx*cond_global.ny);
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global.nx*cond_global.ny)),d_respred,sizeof(float)*cond_global.nx*cond_global.ny,cudaMemcpyDeviceToHost);		
		
	}

	cudaFree(d_y);
	cudaFree(d_respred);
	cudaFree(d_residuals);
	cudaFree(d_uncond);	
	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	*ret_code = 0;
}



void EXPORT conditionalSimRelease(int *ret_code) {
	*ret_code = 1;
	cudaFree(cond_global.d_covinv);
	cufftDestroy(cond_global.plan1);
	cudaFree(cond_global.d_samplexy);
	cudaFree(cond_global.d_sampledata);
	cudaFree(cond_global.d_sampleindices);
	cudaFree(cond_global.d_cov);
	*ret_code = 0;
}

#ifdef __cplusplus
} // extern C
#endif



struct conditioning_state {
	int nx,ny,n,m,k;
	float xmin,xmax,ymin,ymax,dx,dy;
	float range, sill, nugget;
	int blockSize,numBlocks;
	dim3 blockSize2, numBlocks2;
	dim3 blockSize2d;
	dim3 blockCount2d;
	dim3 blockSize1d;
	dim3 blockCount1d;
	dim3 blockSizeSamples;
	dim3 blockCountSamples;
	dim3 blockSizeSamplesPlus1;
	dim3 blockCountSamplesPlus1;
	int numSrc; 
	float2 *d_samplexy; 
	float2 *d_sampleindices; 
	float *d_sampledata; 
	float *d_covinv; 
	float *d_uncond; // Unconditional realizations
	int covmodel;

} conditioning_global;


#ifdef __cplusplus
extern "C" {
#endif


	void EXPORT conditioningInit(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, float *p_sill, float *p_range, float *p_nugget, float *p_srcXY, float *p_srcData, int *p_numSrc, float *p_cov_inv, int *p_k, float *p_uncond, int *p_covmodel, int *ret_code) {
		*ret_code = 1;
		cudaError_t cudaStatus;
		cublasInit();

		conditioning_global.nx= *p_nx; // Number of cols
		conditioning_global.ny= *p_ny; // Number of rows
		conditioning_global.n= 2*conditioning_global.nx; // Number of cols
		conditioning_global.m= 2*conditioning_global.ny; // Number of rows
		conditioning_global.dx = (*p_xmax - *p_xmin) / conditioning_global.nx;
		conditioning_global.dy = (*p_ymax - *p_ymin) / conditioning_global.ny;
		conditioning_global.numSrc = *p_numSrc;
		conditioning_global.xmin = *p_xmin;
		conditioning_global.xmax = *p_xmax;
		conditioning_global.ymin = *p_ymin;
		conditioning_global.ymax = *p_ymax;
		conditioning_global.range = *p_range;
		conditioning_global.sill = *p_sill;
		conditioning_global.nugget = *p_nugget;
		conditioning_global.k = *p_k;
		conditioning_global.covmodel = *p_covmodel;

		// 1d cuda grid
		conditioning_global.blockSize1d = dim3(256);
		conditioning_global.blockCount1d = dim3(conditioning_global.n*conditioning_global.m / conditioning_global.blockSize1d.x);
		if (conditioning_global.n * conditioning_global.m % conditioning_global.blockSize1d.x  != 0) ++conditioning_global.blockCount1d.x;
	
		// 2d cuda grid
		conditioning_global.blockSize2d = dim3(16,16);
		conditioning_global.blockCount2d = dim3(conditioning_global.n / conditioning_global.blockSize2d.x, conditioning_global.m / conditioning_global.blockSize2d.y);
		if (conditioning_global.n % conditioning_global.blockSize2d.x != 0) ++conditioning_global.blockCount2d.x;
		if (conditioning_global.m % conditioning_global.blockSize2d.y != 0) ++conditioning_global.blockCount2d.y;

		// 1d cuda grid der samples
		conditioning_global.blockSizeSamples = dim3(256);
		conditioning_global.blockCountSamples = dim3(conditioning_global.numSrc / conditioning_global.blockSizeSamples.x);
		if (conditioning_global.numSrc % conditioning_global.blockSizeSamples.x !=0) ++conditioning_global.blockCountSamples.x;

		// Copy samples to gpu
		cudaStatus = cudaMalloc((void**)&conditioning_global.d_samplexy,sizeof(float2)* conditioning_global.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaStatus = cudaMalloc((void**)&conditioning_global.d_sampleindices,sizeof(float2)*conditioning_global.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaStatus = cudaMalloc((void**)&conditioning_global.d_sampledata,sizeof(float)*conditioning_global.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaMemcpy(conditioning_global.d_samplexy,p_srcXY,sizeof(float2)* conditioning_global.numSrc,cudaMemcpyHostToDevice);
		cudaMemcpy(conditioning_global.d_sampledata,p_srcData,sizeof(float)*conditioning_global.numSrc,cudaMemcpyHostToDevice);
	

		
		// Overlay samples to grid and save resulting subpixel grind indices
		overlay<<<conditioning_global.blockCountSamples,conditioning_global.blockSizeSamples>>>(conditioning_global.d_sampleindices,conditioning_global.d_samplexy,*p_xmin,conditioning_global.dx,*p_ymax,conditioning_global.dy, conditioning_global.numSrc);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay!\n", cudaStatus);	
		// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access	
		
		// Copy inverse sample cov matrix to gpu
		cudaMalloc((void**)&conditioning_global.d_covinv,sizeof(float) * (conditioning_global.numSrc + 1) * (conditioning_global.numSrc + 1));
		cudaMemcpy(conditioning_global.d_covinv,p_cov_inv,sizeof(float) * (conditioning_global.numSrc + 1) * (conditioning_global.numSrc + 1),cudaMemcpyHostToDevice);


		// Copy given unconditional realizations to gpu
		int size = sizeof(float) * conditioning_global.nx * conditioning_global.ny * conditioning_global.k;
		cudaMalloc((void**)&conditioning_global.d_uncond,size);
		cudaMemcpy(conditioning_global.d_uncond,p_uncond,size,cudaMemcpyHostToDevice);
		*ret_code = 0;
	}


	// Generates unconditional realizations
	// p_out = output array of size nx*ny*k * sizeof(float)
	// ret_code = return code: 0=ok
	void EXPORT conditioningRealizations(float *p_out, int *ret_code) {
		*ret_code = 1;
		cudaError_t cudaStatus;
	
			
		float *d_residuals; // residuals of samples data and unconditional realization
		float *d_y; // result vector of kriging equation system
		float *d_respred; // Interpolated grid of residuals
	

		for(int l = 0; l<conditioning_global.k; ++l) {
			
	
			// Compute sample's residuals
			if (l==0) cudaMalloc((void**)&d_residuals,sizeof(float)* (conditioning_global.numSrc + 1));
			residuals<<<conditioning_global.blockCountSamples,conditioning_global.blockSizeSamples>>>(d_residuals,conditioning_global.d_sampledata,&conditioning_global.d_uncond[l*conditioning_global.nx*conditioning_global.ny],conditioning_global.d_sampleindices,conditioning_global.ny,conditioning_global.ny,conditioning_global.numSrc);
			cudaStatus = cudaThreadSynchronize();	
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
			// Residuals are now stored in d_residual		
			
			// Kriging prediction of residuals:	
			// Compute matrix vector product y = d_covinv * d_residuals on gpu using cublas	
			if (l==0) cudaMalloc((void**)&d_y,sizeof(float)*(conditioning_global.numSrc + 1));	
			cublasSgemv('n',conditioning_global.numSrc + 1,conditioning_global.numSrc + 1, 1.0, conditioning_global.d_covinv,conditioning_global.numSrc + 1,d_residuals,1,0.0,d_y,1);
	
			// Krigin prediction
			if (l==0)  cudaMalloc((void**)&d_respred, sizeof(float) * conditioning_global.nx * conditioning_global.ny);
			dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
			dim3 blockCntKrige = dim3((conditioning_global.nx*conditioning_global.ny) / blockSizeKrige.x);
			if ((conditioning_global.nx*conditioning_global.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;		
			switch(conditioning_global.covmodel) {
			case EXP:
				krigingExpKernel<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global.d_samplexy,conditioning_global.xmin,conditioning_global.dx,conditioning_global.ymin,conditioning_global.dy,d_y,conditioning_global.range,conditioning_global.sill,conditioning_global.nugget,conditioning_global.numSrc,conditioning_global.nx,conditioning_global.ny);
				break;
			case GAU:
				krigingGauKernel<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global.d_samplexy,conditioning_global.xmin,conditioning_global.dx,conditioning_global.ymin,conditioning_global.dy,d_y,conditioning_global.range,conditioning_global.sill,conditioning_global.nugget,conditioning_global.numSrc,conditioning_global.nx,conditioning_global.ny);
				break;
			case SPH:
				krigingSphKernel<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global.d_samplexy,conditioning_global.xmin,conditioning_global.dx,conditioning_global.ymin,conditioning_global.dy,d_y,conditioning_global.range,conditioning_global.sill,conditioning_global.nugget,conditioning_global.numSrc,conditioning_global.nx,conditioning_global.ny);
				break;
			}
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel!\n", cudaStatus);
		
	
			// Add result to unconditional realization
			dim3 blockSizeCond = dim3(256);
			dim3 blockCntCond = dim3(conditioning_global.nx*conditioning_global.ny/ blockSizeCond.x);
			if (conditioning_global.nx*conditioning_global.ny % blockSizeCond.x != 0) ++blockSizeCond.x;
			addResSim<<<blockCntCond,blockSizeCond>>>(d_respred, &conditioning_global.d_uncond[l*conditioning_global.nx*conditioning_global.ny], conditioning_global.nx*conditioning_global.ny);
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim!\n", cudaStatus);
		
			// Write result to R
			cudaMemcpy((p_out + l*(conditioning_global.nx*conditioning_global.ny)),d_respred,sizeof(float)*conditioning_global.nx*conditioning_global.ny,cudaMemcpyDeviceToHost);		
		
		}

		cudaFree(d_y);
		cudaFree(d_respred);
		cudaFree(d_residuals);
		*ret_code = 0;
	}

	void EXPORT conditioningRelease(int *ret_code) {
		*ret_code = 1;
		cudaFree(conditioning_global.d_uncond);
		cudaFree(conditioning_global.d_covinv);
		cudaFree(conditioning_global.d_samplexy);
		cudaFree(conditioning_global.d_sampledata);
		cudaFree(conditioning_global.d_sampleindices);
		*ret_code = 0;
	}


#ifdef __cplusplus
}
#endif



/*******************************************************************************************
** TESTING  ********************************************************************************
********************************************************************************************/
int main()
{	
    
	int count = -1;
	cudaGetDeviceCount(&count);
	std::cout << "Anzahl devices: " << count << "\n";

	int result = -1;
	init(&result);

		
	// Test conditional simulation
	{
		float xmin = 0, xmax = 100, ymin = 0, ymax = 100;
		int nx = 100, ny = 100;
		int covmodel = EXP;
		float nugget=0, range=0.1, sill = 1;
		int k = 5;

		int numSrc = 10;
		float srcxy[] = {64.56559,83.00241,56.55997,66.50534,31.32781,40.72709,39.21148,57.04371,63.70436,69.80745,49.43141,11.15070,17.64743,62.95207,65.10820,83.82076,74.04069,60.73463,74.86754,73.68782};
		float srcdata[] = {98.98939,103.25515,100.31433,108.50051,78.91263,117.80599,104.18466,105.06300,109.77029,102.14903}; 
		
		float covinv[] = {0.31147267,-0.01321620,-0.01345967,-0.01345882,-0.01339282
						,-0.01346126,-0.01346104,-0.20407215,-0.01342644,-0.01352428
						,0.06730629,-0.01321620,0.17913993,-0.02134759,-0.02135648
						,-0.02474149,-0.02135012,-0.02134977,-0.01323224,-0.02131023
						,-0.02123580,0.10675059,-0.01345967,-0.02134759,0.17823321
						,-0.02178865,-0.02124115,-0.02176937,-0.02176945,-0.01349232
						,-0.02171229,-0.02165272,0.10884684,-0.01345882,-0.02135648
						,-0.02178865,0.17823595,-0.02123981,-0.02176800,-0.02177044
						,-0.01349147,-0.02171093,-0.02165136,0.10883999,-0.01339282
						,-0.02474149,-0.02124115,-0.02123981,0.17935024,-0.02124366
						,-0.02124332,-0.01318273,-0.02139313,-0.02167214,0.10621829
						,-0.01346126,-0.02135012,-0.02176937,-0.02176800,-0.02124366
						,0.17822806,-0.02177159,-0.01349392,-0.02171486,-0.02165528
						,0.10885972,-0.01346104,-0.02134977,-0.02176945,-0.02177044
						,-0.02124332,-0.02177159,0.17822876,-0.01349370,-0.02171451
						,-0.02165493,0.10885797,-0.20407215,-0.01323224,-0.01349232
						,-0.01349147,-0.01318273,-0.01349392,-0.01349370,0.31143185
						,-0.01345864,-0.01351466,0.06746960,-0.01342644,-0.02131023
						,-0.02171229,-0.02171093,-0.02139313,-0.02171486,-0.02171451
						,-0.01345864,0.17834275,-0.02190172,0.10857429,-0.01352428
						,-0.02123580,-0.02165272,-0.02165136,-0.02167214,-0.02165528
						,-0.02165493,-0.01351466,-0.02190172,0.17846290,0.10827642
						,0.06730629,0.10675059,0.10884684,0.10883999,0.10621829
						,0.10885972,0.10885797,0.06746960,0.10857429,0.10827642,-0.54429861};

		
		float *cond = (float*)malloc(sizeof(float)*nx*ny*k);
		int retcode = 0;
		conditionalSimInit(&xmin,&xmax,&nx,&ymin,&ymax,&ny,&sill,&range,&nugget,srcxy,srcdata,&numSrc,covinv, &covmodel, &retcode);
		printf("Errorcode: %i\n",retcode);
		conditionalSimRealizations(cond,&k,&retcode);
		conditionalSimRelease(&retcode);

		// write results to csv file for testing purpose
		for (int l=0; l<k; ++l) {
			std::stringstream ss;
			ss << "C:\\fft\\real" << l << ".csv";
			writeCSVMatrix(ss.str().c_str(),cond + l*nx*ny,nx,ny);		
		}
		free(cond);
	}

	system("PAUSE");
}