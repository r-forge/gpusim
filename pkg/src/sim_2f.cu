/**
* sim.cu C and CUDA Interface for gpusim R package
* Author: Marius Appel - marius.appel@uni-muenster.de
*
* TODO: 
*	- Split into several files
*	- Add functions to minimize redundant code
*	- introduce debug args for simpler bug finding
*
*	14.02.2012
**/


#include "utils.h"




/*******************************************************************************************
** GPU KERNELS *****************************************************************************
********************************************************************************************/

// Covariance functions
__device__ float covExpKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/range)));
}


__device__ float covExpAnisKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget, float alpha, float afac1) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	
	temp = dx * cosf(alpha) + dy * sinf(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sinf(alpha)) + dy * cosf(alpha));
	dist += temp * temp;
	dist = sqrtf(dist);

	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/range)));
}



__device__ float covGauKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist2 = (ax-bx)*(ax-bx)+(ay-by)*(ay-by);
	return ((dist2 == 0.0f)? (nugget + sill) : (sill*expf(-dist2/(range*range))));
}


__device__ float covGauAnisKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget, float alpha, float afac1) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	
	temp = dx * cosf(alpha) + dy * sinf(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sinf(alpha)) + dy * cosf(alpha));
	dist += temp * temp;
	//dist = sqrtf(dist);

	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/(range*range))));
}



__device__ float covSphKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0f - (((3.0f*dist) / (2.0f*range)) - ((dist * dist * dist) / (2.0f * range * range * range)) ));	
	return 0.0f; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}

__device__ float covSphAnisKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget, float alpha, float afac1) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	
	temp = dx * cosf(alpha) + dy * sinf(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sinf(alpha)) + dy * cosf(alpha));
	dist += temp * temp;
	dist = sqrtf(dist);

	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0f - (((3.0f*dist) / (2.0f*range)) - ((dist * dist * dist) / (2.0f * range * range * range)) ));	
	return 0.0f; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range

}




__device__ float covMat3Kernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*expf(-SQRT3*dist/range)));
}
__device__ float covMat3AnisKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget, float alpha, float afac1) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	
	temp = dx * cosf(alpha) + dy * sinf(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sinf(alpha)) + dy * cosf(alpha));
	dist += temp * temp;
	dist = sqrtf(dist);

	return ((dist == 0.0f)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*expf(-SQRT3*dist/range)));
}





__device__ float covMat5Kernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * expf(-SQRT5*dist/range)));
}

__device__ float covMat5AnisKernel_2f(float ax, float ay, float bx, float by, float sill, float range, float nugget, float alpha, float afac1) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	
	temp = dx * cosf(alpha) + dy * sinf(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sinf(alpha)) + dy * cosf(alpha));
	dist += temp * temp;
	dist = sqrtf(dist);

	return ((dist == 0.0f)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * expf(-SQRT5*dist/range)));
}




// Converts real float array into cufftComplex array
__global__ void realToComplexKernel_2f(cufftComplex *c, float* r, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		c[i].x = r[i];
		c[i].y = 0.0f;
	}
}




// Gets relevant real parts of 2nx x 2ny grid and devides it by div
__global__ void ReDiv_2f(float *out, cufftComplex *c, float div,int nx, int ny, int M) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//if (col < nx && row < ny) out[col*ny+row] = c[col*2*ny+row].x / div; 
	if (col < nx && row < ny) out[row*nx+col] = c[row*M+col].x / div; 
}









// Covariance sampling of a regular grid
__global__ void sampleCovKernel_2f(cufftComplex *trickgrid, cufftComplex *grid, cufftComplex* cov, float xc, float yc, int model, float sill, float range, float nugget, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		switch (model) {
		case EXP:
			cov[row*n+col].x = covExpKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case GAU:
			cov[row*n+col].x = covGauKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case SPH:
			cov[row*n+col].x = covSphKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case MAT3:
			cov[row*n+col].x = covMat3Kernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case MAT5:
			cov[row*n+col].x = covMat5Kernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		}	
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


__global__ void sampleCovAnisKernel_2f(cufftComplex *trickgrid, cufftComplex *grid, cufftComplex* cov, float xc, float yc, int model, float sill, float range, float nugget, float alpha, float afac1, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		switch (model) {
		case EXP:
			cov[row*n+col].x = covExpAnisKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case GAU:
			cov[row*n+col].x = covGauAnisKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case SPH:
			cov[row*n+col].x = covSphAnisKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case MAT3:
			cov[row*n+col].x = covMat3AnisKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case MAT5:
			cov[row*n+col].x = covMat5AnisKernel_2f(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		}	
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
__global__ void multKernel_2f(cufftComplex *fftgrid, int n, int m) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	fftgrid[i].x = fftgrid[i].x*n*m;
	fftgrid[i].y = fftgrid[i].y*n*m;
}


// Devides spectral grid elementwise by fftgrid
__global__ void divideSpectrumKernel_2f(cufftComplex *spectrum, cufftComplex *fftgrid) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float a = spectrum[i].x;
	float b = spectrum[i].y;
	float c = fftgrid[i].x;
	float d = fftgrid[i].y;
	spectrum[i].x = (a*c+b*d)/(c*c+d*d);
	spectrum[i].y = (b*c-a*d)/(c*c+d*d);
}





// Element-wise sqrt from spectral grid
__global__ void sqrtKernel_2f(cufftComplex *spectrum) {
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
__global__ void elementProduct_2f(cufftComplex *c, cufftComplex *a, cufftComplex *b, int n) {
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



__global__ void krigingKernel_2f(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  int model, float range, float sill, float nugget, int numSrc, int nx, int ny)
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
					switch (model) {
					case EXP:
						sum += covExpKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case GAU:
						sum += covGauKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case SPH:
						sum += covSphKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}


__global__ void krigingAnisKernel_2f(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  int model, float range, float sill, float nugget, float alpha, float afac1, int numSrc, int nx, int ny)
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
					switch (model) {
					case EXP:
						sum += covExpAnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case GAU:
						sum += covGauAnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}







/// Simple Kriging prediction at a regular grid with given samples for conditioning
#ifndef BLOCK_SIZE_KRIGE1
#define BLOCK_SIZE_KRIGE1 256
#endif



__global__ void krigingSimpleKernel_2f(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  int model, float range, float sill, float nugget, int numSrc, int nx, int ny, float mean)
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
					switch (model) {
					case EXP:
						sum += covExpKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case GAU:
						sum += covGauKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case SPH:
						sum += covSphKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + mean;	
}


__global__ void krigingSimpleAnisKernel_2f(float *prediction, float2 *srcXY, float xmin, float dx, float ymin, float dy,  float *y,  int model, float range, float sill, float nugget, float alpha, float afac1, int numSrc, int nx, int ny, float mean)
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
					switch (model) {
					case EXP:
						sum += covExpAnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case GAU:
						sum += covGauAnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_2f(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + mean;	
}







// Adds interpolated residuals element-wise to the unconditional grid. Residual grid will be overwritten in-place
__global__ void addResSim_2f(float *res, float *uncond, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id];
}

__global__ void addResSimMean_2f(float *res, float *uncond, int n, float mean) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id] + mean;
}


// Overlay of samples and regular grid. For each sample, the underlying grid cell in subpixel accuracy is returned, so that a following bilinear 
// interpolation is possible
__global__ void overlay_2f(float2 *out, float2 *xy, float grid_minx, float grid_dx, float grid_maxy, float grid_dy, int numPoints) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPoints) {
		out[i].x = (xy[i].x - grid_minx)/grid_dx;
		out[i].y = (grid_maxy - xy[i].y)/grid_dy;
	}
}


// Calculates residuals of samples and an unconditional realization. Uses bilinear interpolation based on the sample's position in grid
__global__ void residualsOrdinary_2f(float* res, float *srcdata, float *uncond_grid, float2 *indices, int nx, int ny, int numPoints) {
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



// Calculates residuals of samples and an unconditional realization. Uses bilinear interpolation based on the sample's position in grid
__global__ void residualsSimple_2f(float* res, float *srcdata, float *uncond_grid, float2 *indices, int nx, int ny, int numPoints, float mu) {
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
		res[id] = srcdata[id] - mu - ((1-y) * ((1-x) * uncond_grid[row * nx + col] + x * uncond_grid[row * nx + col + 1]) + 
								  y * ((1-x) * uncond_grid[(row+1) * nx + col] + x * uncond_grid[(row+1) * nx + col + 1]));
	}		
}







/*******************************************************************************************
** UNCONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/

// global variables for unconditional simulation. These data are needed in the preprocessing as well as in generating realizations
struct uncond_state_2f {
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
} uncond_global_2f;




#ifdef __cplusplus
extern "C" {
#endif


void EXPORT unconditionalSimInit_2f(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, 
									float *p_sill, float *p_range, float *p_nugget, int *p_covmodel, float *p_anis_direction, 
									float *p_anis_ratio, int *do_check, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	
	uncond_global_2f.nx= *p_nx; // Number of cols
	uncond_global_2f.ny= *p_ny; // Number of rows
	uncond_global_2f.n= 2*uncond_global_2f.nx; // Number of cols
	uncond_global_2f.m= 2*uncond_global_2f.ny; // Number of rows
	//uncond_global_2f.n = ceil2(2*uncond_global_2f.nx); /// 
	//uncond_global_2f.m = ceil2(2*uncond_global_2f.ny); /// 
	uncond_global_2f.dx = (*p_xmax - *p_xmin) / (uncond_global_2f.nx-1);
	uncond_global_2f.dy = (*p_ymax - *p_ymin) / (uncond_global_2f.ny-1);
	
	// 1d cuda grid
	uncond_global_2f.blockSize1d = dim3(256);
	uncond_global_2f.blockCount1d = dim3(uncond_global_2f.n*uncond_global_2f.m / uncond_global_2f.blockSize1d.x);
	if (uncond_global_2f.n * uncond_global_2f.m % uncond_global_2f.blockSize1d.x  != 0) ++uncond_global_2f.blockCount1d.x;
	
	// 2d cuda grid
	uncond_global_2f.blockSize2d = dim3(16,16);
	uncond_global_2f.blockCount2d = dim3(uncond_global_2f.n / uncond_global_2f.blockSize2d.x, uncond_global_2f.m / uncond_global_2f.blockSize2d.y);
	if (uncond_global_2f.n % uncond_global_2f.blockSize2d.x != 0) ++uncond_global_2f.blockCount2d.x;
	if (uncond_global_2f.m % uncond_global_2f.blockSize2d.y != 0) ++uncond_global_2f.blockCount2d.y;

	
	//cufftPlan2d(&uncond_global_2f.plan1, uncond_global_2f.n, uncond_global_2f.m, CUFFT_C2C); 
	cufftPlan2d(&uncond_global_2f.plan1, uncond_global_2f.m, uncond_global_2f.n, CUFFT_C2C); 

	
	// build grid (ROW MAJOR)
	cufftComplex *h_grid_c = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.m*uncond_global_2f.n);
	for (int i=0; i<uncond_global_2f.n; ++i) { // i =  col index
		for (int j=0; j<uncond_global_2f.m; ++j) { // j = row index 
			h_grid_c[j*uncond_global_2f.n+i].x = *p_xmin + i * uncond_global_2f.dx; 
			//h_grid_c[j*uncond_global_2f.n+i].y = *p_ymin + (j+1) * uncond_global_2f.dy;  
			h_grid_c[j*uncond_global_2f.n+i].y = *p_ymin + (uncond_global_2f.m-1-j)* uncond_global_2f.dy;			
		}
	}
	
	
	float xc = *p_xmin + (uncond_global_2f.dx*uncond_global_2f.n)/2;
	float yc = *p_ymin +(uncond_global_2f.dy*uncond_global_2f.m)/2;
	float sill = *p_sill;
	float range = *p_range;
	float nugget = *p_nugget;
	bool isotropic = (*p_anis_ratio == 1.0);
	float afac1 = 1.0/(*p_anis_ratio);
	float alpha = (90.0 - *p_anis_direction) * (PI / 180.0);
	cufftComplex *d_grid;
	
	// Array for grid
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
	// Array for cov grid
	cudaStatus = cudaMalloc((void**)&uncond_global_2f.d_cov,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);

	// Sample covariance and generate "trick" grid
	cufftComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
	
	// copy grid to GPU
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, uncond_global_2f.n*uncond_global_2f.m*sizeof(cufftComplex),cudaMemcpyHostToDevice);
	
	if (isotropic) {
		sampleCovKernel_2f<<<uncond_global_2f.blockCount2d, uncond_global_2f.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global_2f.d_cov, xc, yc,*p_covmodel, sill, range,nugget,uncond_global_2f.n,uncond_global_2f.m);
	}
	else {	
		sampleCovAnisKernel_2f<<<uncond_global_2f.blockCount2d, uncond_global_2f.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global_2f.d_cov, xc, yc, *p_covmodel, sill, range,nugget, alpha, afac1, uncond_global_2f.n,uncond_global_2f.m);	
	}
	free(h_grid_c);
	cudaFree(d_grid);


	 
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE COV MATRIX******* ///////
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_2f.d_cov,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\sampleCov.csv",h_cov,uncond_global_2f.m,uncond_global_2f.n);
//		free(h_cov);
//	}
//#endif

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE TRICK GRID ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\trickgrid.csv",h_cov,uncond_global_2f.m,uncond_global_2f.n);
//		free(h_cov);
//	}
//#endif
//


	// Execute 2d FFT of covariance grid in order to get the spectral representation 
	cufftExecC2C(uncond_global_2f.plan1, uncond_global_2f.d_cov, uncond_global_2f.d_cov, CUFFT_FORWARD); // in place fft forward

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( COV GRID) ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_2f.d_cov,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftSampleCov.csv",h_cov,uncond_global_2f.m,uncond_global_2f.n);
//		free(h_cov);
//	}
//#endif
//
	cufftExecC2C(uncond_global_2f.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forward
	
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( TRICK GRID) ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftTrickGrid.csv",h_cov,uncond_global_2f.m,uncond_global_2f.n);
//		free(h_cov);
//	}
//#endif
	
	// Multiply fft of "trick" grid with n*m
	multKernel_2f<<<uncond_global_2f.blockCount1d, uncond_global_2f.blockSize1d>>>(d_trick_grid_c, uncond_global_2f.n, uncond_global_2f.m);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel_2f!\n", cudaStatus);	


//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( TRICK GRID) ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftTrickGridTimesNM.csv",h_cov,uncond_global_2f.m,uncond_global_2f.n);
//		free(h_cov);
//	}
//#endif


	// Devide spectral covariance grid by "trick" grid
	divideSpectrumKernel_2f<<<uncond_global_2f.blockCount1d, uncond_global_2f.blockSize1d>>>(uncond_global_2f.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel_2f!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);
	
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( COV GRID) / FFT(TRICKGRID)*N*M ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_2f.d_cov,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftSampleCovByTrickGridNM.csv",h_cov,uncond_global_2f.m,uncond_global_2f.n);
//		free(h_cov);
//	}
//#endif
//


	// Copy to host and check for negative real parts
	if (*do_check) {
		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
		cudaStatus = cudaMemcpy(h_cov,uncond_global_2f.d_cov,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m,cudaMemcpyDeviceToHost);
		for (int i=0; i<uncond_global_2f.n*uncond_global_2f.m; ++i) {
			if (h_cov[i].x < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				free(h_cov);
				cudaFree(uncond_global_2f.d_cov);
				cufftDestroy(uncond_global_2f.plan1);
				return;
			}	
		}
		free(h_cov);
	}

	// Compute sqrt of cov grid
	sqrtKernel_2f<<<uncond_global_2f.blockCount1d, uncond_global_2f.blockSize1d>>>(uncond_global_2f.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel_2f\n", cudaStatus);

}

// Generates unconditional realizations
// p_out = output array of size nx*ny*k * sizeof(float)
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT unconditionalSimRealizations_2f(float *p_out,  int *p_k, int *ret_code)
{
	*ret_code = OK;
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
			cudaStatus = cudaMalloc((void**)&d_rand,sizeof(float)*uncond_global_2f.m*uncond_global_2f.n); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
			// Generate Random Numbers
		curandGenerateNormal(curandGen,d_rand,uncond_global_2f.m*uncond_global_2f.n,0.0,1.0);
		
		if(l==0) {
			cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftComplex) * uncond_global_2f.n * uncond_global_2f.m); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
		// Convert real random numbers to complex numbers
		realToComplexKernel_2f<<< uncond_global_2f.blockCount1d, uncond_global_2f.blockSize1d>>>(d_fftrand, d_rand, uncond_global_2f.n*uncond_global_2f.m);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel_2f!\n", cudaStatus);	

		// Compute 2D FFT of random numbers
		cufftExecC2C(uncond_global_2f.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward
		

		if(l==0) cudaMalloc((void**)&d_amp,sizeof(cufftComplex)*uncond_global_2f.n*uncond_global_2f.m);
		elementProduct_2f<<<uncond_global_2f.blockCount1d, uncond_global_2f.blockSize1d>>>(d_amp, uncond_global_2f.d_cov, d_fftrand, uncond_global_2f.m*uncond_global_2f.n);  
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct_2f!\n", cudaStatus);
		
		cufftExecC2C(uncond_global_2f.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation
		
		if(l==0) cudaMalloc((void**)&d_out,sizeof(float)*uncond_global_2f.nx*uncond_global_2f.ny);
		
		dim3 blockSize2dhalf  = dim3(16,16);
		dim3 blockCount2dhalf = dim3(uncond_global_2f.nx/blockSize2dhalf.x,uncond_global_2f.ny/blockSize2dhalf.y);
		if (uncond_global_2f.nx % blockSize2dhalf.x != 0) ++blockCount2dhalf.x;
		if (uncond_global_2f.ny % blockSize2dhalf.y != 0) ++blockCount2dhalf.y;
		ReDiv_2f<<<blockCount2dhalf, blockSize2dhalf>>>(d_out, d_amp, std::sqrt((float)(uncond_global_2f.n*uncond_global_2f.m)), uncond_global_2f.nx, uncond_global_2f.ny, uncond_global_2f.n);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess) {
			printf("cudaThreadSynchronize returned error code %d after launching ReDiv_2f!\n", cudaStatus);
		}

		cudaMemcpy((p_out + l*(uncond_global_2f.nx*uncond_global_2f.ny)),d_out,sizeof(float)*uncond_global_2f.nx*uncond_global_2f.ny,cudaMemcpyDeviceToHost);					
	}

	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_out);
	curandDestroyGenerator(curandGen);
}


void EXPORT unconditionalSimRelease_2f(int *ret_code) {
	*ret_code = OK;
	cudaFree(uncond_global_2f.d_cov);
	cufftDestroy(uncond_global_2f.plan1);
}



#ifdef __cplusplus
}
#endif



/*******************************************************************************************
** CONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/


// global variables for conditional simulation that are needed both, for initialization as well as for generating realizations
struct cond_state_2f {
	cufftComplex *d_cov; 
	int nx,ny,n,m;
	float xmin,xmax,ymin,ymax,dx,dy;
	float range, sill, nugget;
	float alpha, afac1;
	bool isotropic;
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
	//float *d_covinv; // inverse covariance matrix of samples
	float *d_uncond;
	float *h_uncond; //cpu uncond cache
	bool uncond_gpucache;
	int covmodel;
	int k;
	float mu; // known mean for simple kriging
	int krige_method;
} cond_global_2f;













#ifdef __cplusplus
extern "C" {
#endif



void EXPORT conditionalSimInit_2f(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, float *p_sill, 
								  float *p_range, float *p_nugget, float *p_srcXY, float *p_srcData, int *p_numSrc, int *p_covmodel, 
								  float *p_anis_direction, float *p_anis_ratio, int *do_check, int *krige_method, float *mu, int *uncond_gpucache, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	cublasInit();

	cond_global_2f.nx= *p_nx; // Number of cols
	cond_global_2f.ny= *p_ny; // Number of rows
	cond_global_2f.n= 2*cond_global_2f.nx; // Number of cols
	cond_global_2f.m= 2*cond_global_2f.ny; // Number of rows
	cond_global_2f.dx = (*p_xmax - *p_xmin) / (cond_global_2f.nx - 1);
	cond_global_2f.dy = (*p_ymax - *p_ymin) / (cond_global_2f.ny - 1);
	cond_global_2f.numSrc = *p_numSrc;
	cond_global_2f.xmin = *p_xmin;
	cond_global_2f.xmax = *p_xmax;
	cond_global_2f.ymin = *p_ymin;
	cond_global_2f.ymax = *p_ymax;
	cond_global_2f.range = *p_range;
	cond_global_2f.sill = *p_sill;
	cond_global_2f.nugget = *p_nugget;
	cond_global_2f.covmodel = *p_covmodel;
	cond_global_2f.krige_method = *krige_method;
	if (cond_global_2f.krige_method == SIMPLE)
		cond_global_2f.mu = *mu;
	else cond_global_2f.mu = 0;

	cond_global_2f.isotropic = (*p_anis_ratio == 1.0);
	cond_global_2f.afac1 = 1.0/(*p_anis_ratio);
	cond_global_2f.alpha = (90.0 - *p_anis_direction) * (PI / 180.0);

	cond_global_2f.uncond_gpucache = (*uncond_gpucache != 0);

	// 1d cuda grid
	cond_global_2f.blockSize1d = dim3(256);
	cond_global_2f.blockCount1d = dim3(cond_global_2f.n*cond_global_2f.m / cond_global_2f.blockSize1d.x);
	if (cond_global_2f.n * cond_global_2f.m % cond_global_2f.blockSize1d.x  != 0) ++cond_global_2f.blockCount1d.x;
	
	// 2d cuda grid
	cond_global_2f.blockSize2d = dim3(16,16);
	cond_global_2f.blockCount2d = dim3(cond_global_2f.n / cond_global_2f.blockSize2d.x, cond_global_2f.m / cond_global_2f.blockSize2d.y);
	if (cond_global_2f.n % cond_global_2f.blockSize2d.x != 0) ++cond_global_2f.blockCount2d.x;
	if (cond_global_2f.m % cond_global_2f.blockSize2d.y != 0) ++cond_global_2f.blockCount2d.y;

	// 1d cuda grid for samples
	cond_global_2f.blockSizeSamples = dim3(256);
	cond_global_2f.blockCountSamples = dim3(cond_global_2f.numSrc / cond_global_2f.blockSizeSamples.x);
	if (cond_global_2f.numSrc % cond_global_2f.blockSizeSamples.x !=0) ++cond_global_2f.blockCountSamples.x;

	// Setup fft
	//cufftPlan2d(&cond_global_2f.plan1, cond_global_2f.n, cond_global_2f.m, CUFFT_C2C); // n und m vertauscht weil col major grid
	cufftPlan2d(&cond_global_2f.plan1, cond_global_2f.m, cond_global_2f.n, CUFFT_C2C); // n und m vertauscht weil col major grid

	
	
	// Build grid (ROW MAJOR)
	cufftComplex *h_grid_c = (cufftComplex*)malloc(sizeof(cufftComplex)*cond_global_2f.m*cond_global_2f.n);
	for (int i=0; i<cond_global_2f.n; ++i) { // i = col index
		for (int j=0; j<cond_global_2f.m; ++j) { // j = row index
			h_grid_c[j*cond_global_2f.n+i].x = *p_xmin + i * cond_global_2f.dx; 
			//h_grid_c[j*cond_global_2f.n+i].y = *p_ymin + (j+1) * cond_global_2f.dy; 
			h_grid_c[j*cond_global_2f.n+i].y = *p_ymin + (cond_global_2f.m-1-j)* cond_global_2f.dy;
		}
	}

	
	float xc = *p_xmin + (cond_global_2f.dx*cond_global_2f.n)/2;
	float yc = *p_ymin + (cond_global_2f.dy*cond_global_2f.m)/2;
	cufftComplex *d_grid;
	
	// Allocate grid and cov arrays on GPU
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(cufftComplex)*cond_global_2f.n*cond_global_2f.m);
	cudaStatus = cudaMalloc((void**)&cond_global_2f.d_cov,sizeof(cufftComplex)*cond_global_2f.n*cond_global_2f.m);

	// Sample covariance and generate "trick" grid
	cufftComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftComplex)*cond_global_2f.n*cond_global_2f.m);
	
	// Copy grid to gpu
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, cond_global_2f.n*cond_global_2f.m*sizeof(cufftComplex),cudaMemcpyHostToDevice);
	

	if (cond_global_2f.isotropic) {
		sampleCovKernel_2f<<<cond_global_2f.blockCount2d, cond_global_2f.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global_2f.d_cov, xc, yc, cond_global_2f.covmodel, cond_global_2f.sill, cond_global_2f.range, cond_global_2f.nugget, cond_global_2f.n,cond_global_2f.m);
	}
	else {
		sampleCovAnisKernel_2f<<<cond_global_2f.blockCount2d, cond_global_2f.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global_2f.d_cov, xc, yc, cond_global_2f.covmodel, cond_global_2f.sill, cond_global_2f.range, cond_global_2f.nugget,cond_global_2f.alpha,cond_global_2f.afac1, cond_global_2f.n,cond_global_2f.m);
	}

	free(h_grid_c);
	cudaFree(d_grid);


	// Compute spectral representation of cov and "trick" grid
	cufftExecC2C(cond_global_2f.plan1, cond_global_2f.d_cov, cond_global_2f.d_cov, CUFFT_FORWARD); // in place fft forward
	cufftExecC2C(cond_global_2f.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forwar

	// Multiplication of fft(trick_grid) with n*m	
	multKernel_2f<<<cond_global_2f.blockCount1d, cond_global_2f.blockSize1d>>>(d_trick_grid_c, cond_global_2f.n, cond_global_2f.m);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel_2f!\n", cudaStatus);	

	// Devide spectral cov grid by fft of "trick" grid
	divideSpectrumKernel_2f<<<cond_global_2f.blockCount1d, cond_global_2f.blockSize1d>>>(cond_global_2f.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel_2f!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	// Copy to host and check for negative real parts
	if (*do_check) {
		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*cond_global_2f.n*cond_global_2f.m);
		cudaStatus = cudaMemcpy(h_cov,cond_global_2f.d_cov,sizeof(cufftComplex)*cond_global_2f.n*cond_global_2f.m,cudaMemcpyDeviceToHost);
		for (int i=0; i<cond_global_2f.n*cond_global_2f.m; ++i) {
			if (h_cov[i].x < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				free(h_cov);
				cudaFree(cond_global_2f.d_cov);
				cufftDestroy(cond_global_2f.plan1);
				return;
			}	
		}
		free(h_cov);
	}

	// Compute sqrt of spectral cov grid
	sqrtKernel_2f<<<cond_global_2f.blockCount1d, cond_global_2f.blockSize1d>>>(cond_global_2f.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel_2f\n", cudaStatus);

	// Copy samples to gpu
	cudaStatus = cudaMalloc((void**)&cond_global_2f.d_samplexy,sizeof(float2)* cond_global_2f.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global_2f.d_sampleindices,sizeof(float2)*cond_global_2f.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global_2f.d_sampledata,sizeof(float)*cond_global_2f.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMemcpy(cond_global_2f.d_samplexy,p_srcXY,sizeof(float2)* cond_global_2f.numSrc,cudaMemcpyHostToDevice);
	cudaMemcpy(cond_global_2f.d_sampledata,p_srcData,sizeof(float)*cond_global_2f.numSrc,cudaMemcpyHostToDevice);
		

	// Overlay samples to grid and save resulting subpixel grind indices
	overlay_2f<<<cond_global_2f.blockCountSamples,cond_global_2f.blockSizeSamples>>>(cond_global_2f.d_sampleindices,cond_global_2f.d_samplexy,*p_xmin,cond_global_2f.dx,*p_ymax,cond_global_2f.dy, cond_global_2f.numSrc);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay_2f!\n", cudaStatus);	
	// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access			

	/* TEST OUTPUT ON HOST */
	/*float2 *h_indices = (float2*)malloc(sizeof(float2)*cond_global_2f.numSrc);
	cudaMemcpy(h_indices,cond_global_2f.d_sampleindices,sizeof(float2)*cond_global_2f.numSrc,cudaMemcpyDeviceToHost);
	for (int i=0;i<cond_global_2f.numSrc;++i) {
		printf("(%.2f,%.2f) -> (%.2f,%.2f)\n",p_srcXY[2*i],p_srcXY[2*i+1],h_indices[i].x, h_indices[i].y);
	}
	free(h_indices);*/
}




// Generates Unconditional Realizations and the residuals of all samples to all realizations 
// p_out = output matrix of residuals, col means number of realization, row represents a sample point
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT conditionalSimUncondResiduals_2f(float *p_out, int *p_k, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	cond_global_2f.k = *p_k;
	
	float *d_rand; // Device Random Numbers
	curandGenerator_t curandGen;
	cufftComplex *d_fftrand;
	cufftComplex* d_amp;	
	float *d_residuals; // residuals of samples and underlying unconditional realization
	
	curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGen,(unsigned long long)(time(NULL)));	
	
	cudaStatus = cudaMalloc((void**)&d_rand,sizeof(float)*cond_global_2f.m*cond_global_2f.n); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftComplex) * cond_global_2f.n * cond_global_2f.m); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMalloc((void**)&d_amp,sizeof(cufftComplex)*cond_global_2f.n*cond_global_2f.m);
	
	// GPU OR CPU CACHING OF UNCOND REALIZATIONS
	if (cond_global_2f.uncond_gpucache) {
		cudaMalloc((void**)&cond_global_2f.d_uncond,sizeof(float)*cond_global_2f.nx*cond_global_2f.ny * cond_global_2f.k); // Hold all uncond realizations in GPU memory
	}
	else {
		cudaMalloc((void**)&cond_global_2f.d_uncond,sizeof(float)*cond_global_2f.nx*cond_global_2f.ny ); // Only one realization held in GPU memory
		cond_global_2f.h_uncond = (float*)malloc(sizeof(float)*cond_global_2f.nx*cond_global_2f.ny*cond_global_2f.k); // Hold all uncond realizations in CPU main memory
	}


	
	if (cond_global_2f.krige_method == ORDINARY) {
		cudaMalloc((void**)&d_residuals,sizeof(float)* (cond_global_2f.numSrc + 1));
	}
	else if (cond_global_2f.krige_method == SIMPLE) {
		cudaMalloc((void**)&d_residuals,sizeof(float)* cond_global_2f.numSrc);
	}
		
	for(int l=0; l<cond_global_2f.k; ++l) {
			
		
		curandGenerateNormal(curandGen,d_rand,cond_global_2f.m*cond_global_2f.n,0.0,1.0);	
		realToComplexKernel_2f<<< cond_global_2f.blockCount1d, cond_global_2f.blockSize1d>>>(d_fftrand, d_rand, cond_global_2f.n*cond_global_2f.m);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel_2f!\n", cudaStatus);	
		cufftExecC2C(cond_global_2f.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward
		cudaStatus = cudaThreadSynchronize();
		
		elementProduct_2f<<<cond_global_2f.blockCount1d, cond_global_2f.blockSize1d>>>(d_amp, cond_global_2f.d_cov, d_fftrand, cond_global_2f.m*cond_global_2f.n);
    
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct_2f!\n", cudaStatus);

		cufftExecC2C(cond_global_2f.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation
	  
		dim3 blockSize2dhalf  = dim3(16,16);
		dim3 blockCount2dhalf = dim3(cond_global_2f.nx/blockSize2dhalf.x,cond_global_2f.ny/blockSize2dhalf.y);
		if (cond_global_2f.nx % blockSize2dhalf.x != 0) ++blockCount2dhalf.x;
		if (cond_global_2f.ny % blockSize2dhalf.y != 0) ++blockCount2dhalf.y;

		if (cond_global_2f.uncond_gpucache) {
			ReDiv_2f<<<blockCount2dhalf, blockSize2dhalf>>>(cond_global_2f.d_uncond + l*cond_global_2f.nx*cond_global_2f.ny, d_amp, std::sqrt((float)(cond_global_2f.n*cond_global_2f.m)), cond_global_2f.nx, cond_global_2f.ny, cond_global_2f.n);
		}
		else {
			ReDiv_2f<<<blockCount2dhalf, blockSize2dhalf>>>(cond_global_2f.d_uncond, d_amp, std::sqrt((float)(cond_global_2f.n*cond_global_2f.m)), cond_global_2f.nx, cond_global_2f.ny, cond_global_2f.n);
		}
		cudaStatus = cudaThreadSynchronize();	
		
		
		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching ReDiv_2f!\n", cudaStatus);
		
		// d_uncond is now a unconditional realization 
		// Compute residuals between samples and d_uncond
		if (cond_global_2f.uncond_gpucache) {
			if (cond_global_2f.krige_method == ORDINARY) {
				residualsOrdinary_2f<<<cond_global_2f.blockCountSamples,cond_global_2f.blockSizeSamples>>>(d_residuals,cond_global_2f.d_sampledata,cond_global_2f.d_uncond+l*(cond_global_2f.nx*cond_global_2f.ny),cond_global_2f.d_sampleindices,cond_global_2f.nx,cond_global_2f.ny,cond_global_2f.numSrc);
			}
			else if (cond_global_2f.krige_method == SIMPLE) {
				residualsSimple_2f<<<cond_global_2f.blockCountSamples,cond_global_2f.blockSizeSamples>>>(d_residuals,cond_global_2f.d_sampledata,cond_global_2f.d_uncond+l*(cond_global_2f.nx*cond_global_2f.ny),cond_global_2f.d_sampleindices,cond_global_2f.nx,cond_global_2f.ny,cond_global_2f.numSrc, cond_global_2f.mu);
			}
		}
		else {
			if (cond_global_2f.krige_method == ORDINARY) {
				residualsOrdinary_2f<<<cond_global_2f.blockCountSamples,cond_global_2f.blockSizeSamples>>>(d_residuals,cond_global_2f.d_sampledata,cond_global_2f.d_uncond,cond_global_2f.d_sampleindices,cond_global_2f.nx,cond_global_2f.ny,cond_global_2f.numSrc);
			}
			else if (cond_global_2f.krige_method == SIMPLE) {
				residualsSimple_2f<<<cond_global_2f.blockCountSamples,cond_global_2f.blockSizeSamples>>>(d_residuals,cond_global_2f.d_sampledata,cond_global_2f.d_uncond,cond_global_2f.d_sampleindices,cond_global_2f.nx,cond_global_2f.ny,cond_global_2f.numSrc, cond_global_2f.mu);
			}
		}


		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
	

		// Copy residuals to R, col major...
		if (cond_global_2f.krige_method == ORDINARY) {
			cudaMemcpy((p_out + l*(cond_global_2f.numSrc + 1)),d_residuals,sizeof(float)* (cond_global_2f.numSrc + 1),cudaMemcpyDeviceToHost);	
		}
		else if (cond_global_2f.krige_method == SIMPLE) {
			cudaMemcpy(p_out + l*cond_global_2f.numSrc,d_residuals,sizeof(float) * cond_global_2f.numSrc,cudaMemcpyDeviceToHost);	
		}

		// if needed, cache uncond realizations on cpu
		if (!cond_global_2f.uncond_gpucache) {
			cudaMemcpy(cond_global_2f.h_uncond+l*cond_global_2f.nx*cond_global_2f.ny,cond_global_2f.d_uncond, sizeof(float)*cond_global_2f.nx*cond_global_2f.ny,cudaMemcpyDeviceToHost);
		}
	}
	curandDestroyGenerator(curandGen);
	
	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_residuals);
}


void EXPORT conditionalSimKrigeResiduals_2f(float *p_out, float *p_y, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus = cudaSuccess;
	
	float *d_y; // result vector from solving the kriging equation system
	float *d_respred; // interpolated residuals
	cudaMalloc((void**)&d_y, sizeof(float) * (cond_global_2f.numSrc + 1));
	cudaMalloc((void**)&d_respred, sizeof(float) * cond_global_2f.nx * cond_global_2f.ny);
	
	dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
	dim3 blockCntKrige = dim3((cond_global_2f.nx*cond_global_2f.ny) / blockSizeKrige.x);
	if ((cond_global_2f.nx*cond_global_2f.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;
	
	dim3 blockSizeCond = dim3(256);
	dim3 blockCntCond = dim3(cond_global_2f.nx*cond_global_2f.ny/ blockSizeCond.x);
	if (cond_global_2f.nx*cond_global_2f.ny % blockSizeCond.x != 0) ++blockCntCond.x;

	for(int l = 0; l<cond_global_2f.k; ++l) {
						
		
		cudaMemcpy(d_y, p_y + l*(cond_global_2f.numSrc + 1), sizeof(float) * (cond_global_2f.numSrc + 1),cudaMemcpyHostToDevice);		
		
		// Kriging prediction

		if (cond_global_2f.isotropic)
			krigingKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2f.d_samplexy,cond_global_2f.xmin,cond_global_2f.dx,cond_global_2f.ymin,cond_global_2f.dy,d_y,cond_global_2f.covmodel,cond_global_2f.range,cond_global_2f.sill,cond_global_2f.nugget,cond_global_2f.numSrc,cond_global_2f.nx,cond_global_2f.ny);
		else 	
			krigingAnisKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2f.d_samplexy,cond_global_2f.xmin,cond_global_2f.dx,cond_global_2f.ymin,cond_global_2f.dy,d_y,cond_global_2f.covmodel,cond_global_2f.range,cond_global_2f.sill,cond_global_2f.nugget,cond_global_2f.alpha,cond_global_2f.afac1,cond_global_2f.numSrc,cond_global_2f.nx,cond_global_2f.ny);

		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2f!\n", cudaStatus);
		
		// Add result to unconditional realization
		if (cond_global_2f.uncond_gpucache) {
			addResSim_2f<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2f.d_uncond + l*cond_global_2f.nx*cond_global_2f.ny, cond_global_2f.nx*cond_global_2f.ny);
		}
		else {
			cudaMemcpy(cond_global_2f.d_uncond, cond_global_2f.h_uncond+l*cond_global_2f.nx*cond_global_2f.ny, sizeof(float)*cond_global_2f.nx*cond_global_2f.ny,cudaMemcpyHostToDevice);
			addResSim_2f<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2f.d_uncond, cond_global_2f.nx*cond_global_2f.ny);	
		}

		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_2f!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global_2f.nx*cond_global_2f.ny)),d_respred,sizeof(float)*cond_global_2f.nx*cond_global_2f.ny,cudaMemcpyDeviceToHost);		
		
	}
	cudaFree(d_y);
	cudaFree(d_respred);
}




void EXPORT conditionalSimSimpleKrigeResiduals_2f(float *p_out, float *p_y, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus = cudaSuccess;
	
	float *d_y; // result vector from solving the kriging equation system
	float *d_respred; // interpolated residuals
	cudaMalloc((void**)&d_y, sizeof(float) * cond_global_2f.numSrc); // not + 1, no lagrange multiplicator in simple kriging
	cudaMalloc((void**)&d_respred, sizeof(float) * cond_global_2f.nx * cond_global_2f.ny);
	
	dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
	dim3 blockCntKrige = dim3((cond_global_2f.nx*cond_global_2f.ny) / blockSizeKrige.x);
	if ((cond_global_2f.nx*cond_global_2f.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;
	
	dim3 blockSizeCond = dim3(256);
	dim3 blockCntCond = dim3(cond_global_2f.nx*cond_global_2f.ny/ blockSizeCond.x);
	if (cond_global_2f.nx*cond_global_2f.ny % blockSizeCond.x != 0) ++blockCntCond.x;

	for(int l = 0; l<cond_global_2f.k; ++l) {
						
		
		cudaMemcpy(d_y, p_y + l*cond_global_2f.numSrc, sizeof(float) * cond_global_2f.numSrc,cudaMemcpyHostToDevice);	// not + 1, no lagrange multiplicator in simple kriging		
		// Kriging prediction

		if (cond_global_2f.isotropic)
			krigingSimpleKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2f.d_samplexy,cond_global_2f.xmin,cond_global_2f.dx,cond_global_2f.ymin,cond_global_2f.dy,d_y,cond_global_2f.covmodel,cond_global_2f.range,cond_global_2f.sill,cond_global_2f.nugget,cond_global_2f.numSrc,cond_global_2f.nx,cond_global_2f.ny,cond_global_2f.mu);			
		else 	
			krigingSimpleAnisKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2f.d_samplexy,cond_global_2f.xmin,cond_global_2f.dx,cond_global_2f.ymin,cond_global_2f.dy,d_y,cond_global_2f.covmodel,cond_global_2f.range,cond_global_2f.sill,cond_global_2f.nugget,cond_global_2f.alpha , cond_global_2f.afac1 ,cond_global_2f.numSrc,cond_global_2f.nx,cond_global_2f.ny,cond_global_2f.mu);			

		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2f!\n", cudaStatus);
		
		
		// Add result to unconditional realization
		if (cond_global_2f.uncond_gpucache) {
			addResSim_2f<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2f.d_uncond + l*cond_global_2f.nx*cond_global_2f.ny, cond_global_2f.nx*cond_global_2f.ny);
		}
		else {
			cudaMemcpy(cond_global_2f.d_uncond, cond_global_2f.h_uncond+l*cond_global_2f.nx*cond_global_2f.ny, sizeof(float)*cond_global_2f.nx*cond_global_2f.ny,cudaMemcpyHostToDevice);
			addResSim_2f<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2f.d_uncond, cond_global_2f.nx*cond_global_2f.ny);	
		}
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_2f!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global_2f.nx*cond_global_2f.ny)),d_respred,sizeof(float)*cond_global_2f.nx*cond_global_2f.ny,cudaMemcpyDeviceToHost);				
	}
	cudaFree(d_y);
	cudaFree(d_respred);
}


void EXPORT conditionalSimRelease_2f(int *ret_code) {
	*ret_code = OK;
	cufftDestroy(cond_global_2f.plan1);
	cudaFree(cond_global_2f.d_samplexy);
	cudaFree(cond_global_2f.d_sampledata);
	cudaFree(cond_global_2f.d_sampleindices);
	cudaFree(cond_global_2f.d_cov);
	cudaFree(cond_global_2f.d_uncond);

	if (!cond_global_2f.uncond_gpucache) {
		free(cond_global_2f.h_uncond);
	}
}



#ifdef __cplusplus
}
#endif
















struct conditioning_state_2f {
	int nx,ny,n,m,k;
	float xmin,xmax,ymin,ymax,dx,dy;
	float range, sill, nugget;
	bool isotropic;
	float alpha,afac1;

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
	//float *d_covinv; 
	float *d_uncond; // Unconditional realizations
	int covmodel;
	float mu; // known mean for simple kriging
	int krige_method;

} conditioning_global_2f;

#ifdef __cplusplus
extern "C" {
#endif


	void EXPORT conditioningInit_2f(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, float *p_sill, 
									float *p_range, float *p_nugget, float *p_srcXY, float *p_srcData, int *p_numSrc,  int *p_k, 
									float *p_uncond, int *p_covmodel, float *p_anis_direction, float *p_anis_ratio,int *krige_method, 
									float *mu, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus;
		cublasInit();

		conditioning_global_2f.nx= *p_nx; // Number of cols
		conditioning_global_2f.ny= *p_ny; // Number of rows
		conditioning_global_2f.n= 2*conditioning_global_2f.nx; // Number of cols
		conditioning_global_2f.m= 2*conditioning_global_2f.ny; // Number of rows
		conditioning_global_2f.dx = (*p_xmax - *p_xmin) / (conditioning_global_2f.nx-1);
		conditioning_global_2f.dy = (*p_ymax - *p_ymin) / (conditioning_global_2f.ny-1);
		conditioning_global_2f.numSrc = *p_numSrc;
		conditioning_global_2f.xmin = *p_xmin;
		conditioning_global_2f.xmax = *p_xmax;
		conditioning_global_2f.ymin = *p_ymin;
		conditioning_global_2f.ymax = *p_ymax;
		conditioning_global_2f.range = *p_range;
		conditioning_global_2f.sill = *p_sill;
		conditioning_global_2f.nugget = *p_nugget;
		conditioning_global_2f.k = *p_k;
		conditioning_global_2f.covmodel = *p_covmodel;
		conditioning_global_2f.krige_method = *krige_method;
		if (cond_global_2f.krige_method == SIMPLE)
			conditioning_global_2f.mu = *mu;
		else conditioning_global_2f.mu = 0;

		conditioning_global_2f.isotropic = (*p_anis_ratio == 1.0);
		conditioning_global_2f.afac1 = 1/(*p_anis_ratio);
		conditioning_global_2f.alpha = (90.0 - *p_anis_direction) * (PI / 180.0);

		// 1d cuda grid
		conditioning_global_2f.blockSize1d = dim3(256);
		conditioning_global_2f.blockCount1d = dim3(conditioning_global_2f.n*conditioning_global_2f.m / conditioning_global_2f.blockSize1d.x);
		if (conditioning_global_2f.n * conditioning_global_2f.m % conditioning_global_2f.blockSize1d.x  != 0) ++conditioning_global_2f.blockCount1d.x;
	
		// 2d cuda grid
		conditioning_global_2f.blockSize2d = dim3(16,16);
		conditioning_global_2f.blockCount2d = dim3(conditioning_global_2f.n / conditioning_global_2f.blockSize2d.x, conditioning_global_2f.m / conditioning_global_2f.blockSize2d.y);
		if (conditioning_global_2f.n % conditioning_global_2f.blockSize2d.x != 0) ++conditioning_global_2f.blockCount2d.x;
		if (conditioning_global_2f.m % conditioning_global_2f.blockSize2d.y != 0) ++conditioning_global_2f.blockCount2d.y;

		// 1d cuda grid der samples
		conditioning_global_2f.blockSizeSamples = dim3(256);
		conditioning_global_2f.blockCountSamples = dim3(conditioning_global_2f.numSrc / conditioning_global_2f.blockSizeSamples.x);
		if (conditioning_global_2f.numSrc % conditioning_global_2f.blockSizeSamples.x !=0) ++conditioning_global_2f.blockCountSamples.x;

		// Copy samples to gpu
		cudaStatus = cudaMalloc((void**)&conditioning_global_2f.d_samplexy,sizeof(float2)* conditioning_global_2f.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaStatus = cudaMalloc((void**)&conditioning_global_2f.d_sampleindices,sizeof(float2)*conditioning_global_2f.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaStatus = cudaMalloc((void**)&conditioning_global_2f.d_sampledata,sizeof(float)*conditioning_global_2f.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaMemcpy(conditioning_global_2f.d_samplexy,p_srcXY,sizeof(float2)* conditioning_global_2f.numSrc,cudaMemcpyHostToDevice);
		cudaMemcpy(conditioning_global_2f.d_sampledata,p_srcData,sizeof(float)*conditioning_global_2f.numSrc,cudaMemcpyHostToDevice);
	

		
		// Overlay samples to grid and save resulting subpixel grid indices
		overlay_2f<<<conditioning_global_2f.blockCountSamples,conditioning_global_2f.blockSizeSamples>>>(conditioning_global_2f.d_sampleindices,conditioning_global_2f.d_samplexy,*p_xmin,conditioning_global_2f.dx,*p_ymax,conditioning_global_2f.dy, conditioning_global_2f.numSrc);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay_2f!\n", cudaStatus);	
		// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access	
		
		// Copy inverse sample cov matrix to gpu
		//cudaMalloc((void**)&conditioning_global_2f.d_covinv,sizeof(float) * (conditioning_global_2f.numSrc + 1) * (conditioning_global_2f.numSrc + 1));
		//cudaMemcpy(conditioning_global_2f.d_covinv,p_cov_inv,sizeof(float) * (conditioning_global_2f.numSrc + 1) * (conditioning_global_2f.numSrc + 1),cudaMemcpyHostToDevice);


		// Copy given unconditional realizations to gpu
		int size = sizeof(float) * conditioning_global_2f.nx * conditioning_global_2f.ny * conditioning_global_2f.k;
		cudaMalloc((void**)&conditioning_global_2f.d_uncond,size);
		cudaMemcpy(conditioning_global_2f.d_uncond,p_uncond,size,cudaMemcpyHostToDevice);
	}


	// Generates unconditional realizations
	// p_out = output array of size nx*ny*k * sizeof(float)
	// ret_code = return code: 0=ok
	void EXPORT conditioningResiduals_2f(float *p_out, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus;
			
		float *d_residuals; // residuals of sample data and unconditional realization
		
		if (conditioning_global_2f.krige_method == ORDINARY)
			cudaMalloc((void**)&d_residuals,sizeof(float)* (conditioning_global_2f.numSrc + 1));
		else if (conditioning_global_2f.krige_method == SIMPLE)
			cudaMalloc((void**)&d_residuals,sizeof(float)* (conditioning_global_2f.numSrc));

		for(int l = 0; l<conditioning_global_2f.k; ++l) {
			
			// d_uncond is now a unconditional realization 
			// Compute residuals between samples and d_uncond
			if (conditioning_global_2f.krige_method == ORDINARY) {
				residualsOrdinary_2f<<<conditioning_global_2f.blockCountSamples,conditioning_global_2f.blockSizeSamples>>>(d_residuals,conditioning_global_2f.d_sampledata,conditioning_global_2f.d_uncond+l*(conditioning_global_2f.nx*conditioning_global_2f.ny),conditioning_global_2f.d_sampleindices,conditioning_global_2f.nx,conditioning_global_2f.ny,conditioning_global_2f.numSrc);
			}
			else if (cond_global_2f.krige_method == SIMPLE) {
				residualsSimple_2f<<<conditioning_global_2f.blockCountSamples,conditioning_global_2f.blockSizeSamples>>>(d_residuals,conditioning_global_2f.d_sampledata,conditioning_global_2f.d_uncond+l*(conditioning_global_2f.nx*conditioning_global_2f.ny),conditioning_global_2f.d_sampleindices,conditioning_global_2f.nx,conditioning_global_2f.ny,conditioning_global_2f.numSrc,conditioning_global_2f.mu);
			}
			cudaStatus = cudaThreadSynchronize();	
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
	

			// Copy residuals to R, col major...
			if (conditioning_global_2f.krige_method == ORDINARY) {
				cudaMemcpy((p_out + l*(conditioning_global_2f.numSrc + 1)),d_residuals,sizeof(float)* (conditioning_global_2f.numSrc + 1),cudaMemcpyDeviceToHost);	
			}
			else if (conditioning_global_2f.krige_method == SIMPLE) {
				cudaMemcpy(p_out + l*conditioning_global_2f.numSrc,d_residuals,sizeof(float) * conditioning_global_2f.numSrc,cudaMemcpyDeviceToHost);	
			}
		}	
		cudaFree(d_residuals);
	}


	void EXPORT conditioningKrigeResiduals_2f(float *p_out, float *p_y, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus = cudaSuccess;

		float *d_y; // result vector of kriging equation system
		float *d_respred; // Interpolated grid of residuals

		cudaMalloc((void**)&d_y, sizeof(float) * (conditioning_global_2f.numSrc + 1));
		cudaMalloc((void**)&d_respred, sizeof(float) * conditioning_global_2f.nx * conditioning_global_2f.ny);

		dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
		dim3 blockCntKrige = dim3((conditioning_global_2f.nx*conditioning_global_2f.ny) / blockSizeKrige.x);
		if ((conditioning_global_2f.nx*conditioning_global_2f.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;

		dim3 blockSizeCond = dim3(256);
		dim3 blockCntCond = dim3(conditioning_global_2f.nx*conditioning_global_2f.ny/ blockSizeCond.x);
		if (conditioning_global_2f.nx*conditioning_global_2f.ny % blockSizeCond.x != 0) ++blockCntCond.x;

		for(int l = 0; l<cond_global_2f.k; ++l) {
			cudaMemcpy(d_y, p_y + l*(conditioning_global_2f.numSrc + 1), sizeof(float) * (conditioning_global_2f.numSrc + 1),cudaMemcpyHostToDevice);		
					
			if (conditioning_global_2f.isotropic) {
				krigingKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2f.d_samplexy,conditioning_global_2f.xmin,conditioning_global_2f.dx,conditioning_global_2f.ymin,conditioning_global_2f.dy,d_y,conditioning_global_2f.covmodel,conditioning_global_2f.range,conditioning_global_2f.sill,conditioning_global_2f.nugget,conditioning_global_2f.numSrc,conditioning_global_2f.nx,conditioning_global_2f.ny);			
			}
			else {
				krigingAnisKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2f.d_samplexy,conditioning_global_2f.xmin,conditioning_global_2f.dx,conditioning_global_2f.ymin,conditioning_global_2f.dy,d_y,conditioning_global_2f.covmodel,conditioning_global_2f.range,conditioning_global_2f.sill,conditioning_global_2f.nugget,conditioning_global_2f.alpha,conditioning_global_2f.afac1,conditioning_global_2f.numSrc,conditioning_global_2f.nx,conditioning_global_2f.ny);			
			}
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2f!\n", cudaStatus);
		
			// Add result to unconditional realization	
			addResSim_2f<<<blockCntCond,blockSizeCond>>>(d_respred, &conditioning_global_2f.d_uncond[l*conditioning_global_2f.nx*conditioning_global_2f.ny], conditioning_global_2f.nx*conditioning_global_2f.ny);
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_2f!\n", cudaStatus);
		
			// Write result to R
			cudaMemcpy((p_out + l*(conditioning_global_2f.nx*conditioning_global_2f.ny)),d_respred,sizeof(float)*conditioning_global_2f.nx*conditioning_global_2f.ny,cudaMemcpyDeviceToHost);		
		
		}

		cudaFree(d_y);
		cudaFree(d_respred);
	}




	void EXPORT conditioningSimpleKrigeResiduals_2f(float *p_out, float *p_y, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus = cudaSuccess;

		float *d_y; // result vector of kriging equation system
		float *d_respred; // Interpolated grid of residuals

		cudaMalloc((void**)&d_y, sizeof(float) * conditioning_global_2f.numSrc);
		cudaMalloc((void**)&d_respred, sizeof(float) * conditioning_global_2f.nx * conditioning_global_2f.ny);

		dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
		dim3 blockCntKrige = dim3((conditioning_global_2f.nx*conditioning_global_2f.ny) / blockSizeKrige.x);
		if ((conditioning_global_2f.nx*conditioning_global_2f.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;

		dim3 blockSizeCond = dim3(256);
		dim3 blockCntCond = dim3(conditioning_global_2f.nx*conditioning_global_2f.ny/ blockSizeCond.x);
		if (conditioning_global_2f.nx*conditioning_global_2f.ny % blockSizeCond.x != 0) ++blockCntCond.x;

		for(int l = 0; l<cond_global_2f.k; ++l) {
			cudaMemcpy(d_y, p_y + l*conditioning_global_2f.numSrc, sizeof(float) * conditioning_global_2f.numSrc,cudaMemcpyHostToDevice);		
				
			if (conditioning_global_2f.isotropic) {
				krigingSimpleKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2f.d_samplexy,conditioning_global_2f.xmin,conditioning_global_2f.dx,conditioning_global_2f.ymin,conditioning_global_2f.dy,d_y,conditioning_global_2f.covmodel,conditioning_global_2f.range,conditioning_global_2f.sill,conditioning_global_2f.nugget,conditioning_global_2f.numSrc,conditioning_global_2f.nx,conditioning_global_2f.ny,conditioning_global_2f.mu);
			}
			else {
				krigingSimpleAnisKernel_2f<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2f.d_samplexy,conditioning_global_2f.xmin,conditioning_global_2f.dx,conditioning_global_2f.ymin,conditioning_global_2f.dy,d_y,conditioning_global_2f.covmodel,conditioning_global_2f.range,conditioning_global_2f.sill,conditioning_global_2f.nugget,conditioning_global_2f.alpha,conditioning_global_2f.afac1,conditioning_global_2f.numSrc,conditioning_global_2f.nx,conditioning_global_2f.ny,conditioning_global_2f.mu);
			}
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2f!\n", cudaStatus);
		
			// Add result to unconditional realization	
			addResSim_2f<<<blockCntCond,blockSizeCond>>>(d_respred, &conditioning_global_2f.d_uncond[l*conditioning_global_2f.nx*conditioning_global_2f.ny], conditioning_global_2f.nx*conditioning_global_2f.ny);
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_2f!\n", cudaStatus);
		
			// Write result to R
			cudaMemcpy((p_out + l*(conditioning_global_2f.nx*conditioning_global_2f.ny)),d_respred,sizeof(float)*conditioning_global_2f.nx*conditioning_global_2f.ny,cudaMemcpyDeviceToHost);		
		
		}

		cudaFree(d_y);
		cudaFree(d_respred);
	}


	void EXPORT conditioningRelease_2f(int *ret_code) {
		*ret_code = OK;
		cudaFree(conditioning_global_2f.d_uncond);
		cudaFree(conditioning_global_2f.d_samplexy);
		cudaFree(conditioning_global_2f.d_sampledata);
		cudaFree(conditioning_global_2f.d_sampleindices);
	}


#ifdef __cplusplus
}
#endif



