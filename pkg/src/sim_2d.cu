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

__device__ double covExpKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/range)));
}



__device__ double covExpAnisKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/range)));
}





__device__ double covGauKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist2 = (ax-bx)*(ax-bx)+(ay-by)*(ay-by);
	return ((dist2 == 0.0f)? (nugget + sill) : (sill*expf(-dist2/(range*range))));
}



__device__ double covGauAnisKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	//dist = sqrtf(dist);

	return ((dist == 0.0f)? (nugget + sill) : (sill*expf(-dist/(range*range))));
}


__device__ double covSphKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0 - (((3.0*dist) / (2.0*range)) - ((dist * dist * dist) / (2.0 * range * range * range)) ));	
	return 0.0; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}



__device__ double covSphAnisKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0 - (((3.0*dist) / (2.0*range)) - ((dist * dist * dist) / (2.0 * range * range * range)) ));	
	return 0.0; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}


//Maternmodelle ohne Besselfunktionen

__device__ double covMat3Kernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*exp(-SQRT3*dist/range)));
}


__device__ double covMat3AnisKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	return ((dist == 0.0)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*expf(-SQRT3*dist/range)));
}


__device__ double covMat5Kernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0f)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * exp(-SQRT5*dist/range)));
}


__device__ double covMat5AnisKernel_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	return ((dist == 0.0)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * expf(-SQRT5*dist/range)));
}




// Converts real double array into cufftDoubleComplex array
__global__ void realToComplexKernel_2d(cufftDoubleComplex *c, double* r, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		c[i].x = r[i];
		c[i].y = 0.0f;
	}
}

__global__ void ReDiv_2d(double *out, cufftDoubleComplex *c, double div,int nx, int ny, int M) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//if (col < nx && row < ny) out[col*ny+row] = c[col*2*ny+row].x / div; 
	if (col < nx && row < ny) out[row*nx+col] = c[row*M+col].x / div; 
}





// Covariance sampling of a regular grid
__global__ void sampleCovKernel_2d(cufftDoubleComplex *trickgrid, cufftDoubleComplex *grid, cufftDoubleComplex* cov, double xc, double yc, int model, double sill, double range, double nugget, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		switch (model) {
		case EXP:
			cov[row*n+col].x = covExpKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case GAU:
			cov[row*n+col].x = covGauKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case SPH:
			cov[row*n+col].x = covSphKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case MAT3:
			cov[row*n+col].x = covMat3Kernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
			break;
		case MAT5:
			cov[row*n+col].x = covMat5Kernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget);
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




__global__ void sampleCovAnisKernel_2d(cufftDoubleComplex *trickgrid, cufftDoubleComplex *grid, cufftDoubleComplex* cov, double xc, double yc, int model, double sill, double range, double nugget, double alpha, double afac1, int n, int m) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < n && row < m) {
		switch (model) {
		case EXP:
			cov[row*n+col].x = covExpAnisKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case GAU:
			cov[row*n+col].x = covGauAnisKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case SPH:
			cov[row*n+col].x = covSphAnisKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case MAT3:
			cov[row*n+col].x = covMat3AnisKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
			break;
		case MAT5:
			cov[row*n+col].x = covMat5AnisKernel_2d(grid[row*n+col].x,grid[row*n+col].y,xc,yc,sill,range,nugget,alpha,afac1);
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




__global__ void multKernel_2d(cufftDoubleComplex *fftgrid, int n, int m) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	fftgrid[i].x = fftgrid[i].x*n*m;
	fftgrid[i].y = fftgrid[i].y*n*m;
}


// Devides spectral grid elementwise by fftgrid
__global__ void divideSpectrumKernel_2d(cufftDoubleComplex *spectrum, cufftDoubleComplex *fftgrid) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double a = spectrum[i].x;
	double b = spectrum[i].y;
	double c = fftgrid[i].x;
	double d = fftgrid[i].y;
	spectrum[i].x = (a*c+b*d)/(c*c+d*d);
	spectrum[i].y = (b*c-a*d)/(c*c+d*d);
}



// Element-wise sqrt from spectral grid
__global__ void sqrtKernel_2d(cufftDoubleComplex *spectrum) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double re = spectrum[i].x;
	double im = spectrum[i].y;
	double sill = 0;
	double d = sqrt(re*re+im*im);
	double dsqrt = sqrt(d);
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
__global__ void elementProduct_2d(cufftDoubleComplex *c, cufftDoubleComplex *a, cufftDoubleComplex *b, int n) {
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


//KRIGING KERNELS



__global__ void krigingKernel_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, int numSrc, int nx, int ny)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	double sum=0.0f;
	double yr_x, yr_y;
	
	__shared__ double qs[BLOCK_SIZE_KRIGE1];
	__shared__ double Xs[BLOCK_SIZE_KRIGE1][2];

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
						sum += covExpKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case GAU:
						sum += covGauKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case SPH:
						sum += covSphKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}


__global__ void krigingAnisKernel_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, double alpha, double afac1, int numSrc, int nx, int ny)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	double sum=0.0f;
	double yr_x, yr_y;
	
	__shared__ double qs[BLOCK_SIZE_KRIGE1];
	__shared__ double Xs[BLOCK_SIZE_KRIGE1][2];

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
						sum += covExpAnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case GAU:
						sum += covGauAnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
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










__global__ void krigingSimpleKernel_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, int numSrc, int nx, int ny, double mean)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	double sum=0.0f;
	double yr_x, yr_y;
	
	__shared__ double qs[BLOCK_SIZE_KRIGE1];
	__shared__ double Xs[BLOCK_SIZE_KRIGE1][2];

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
						sum += covExpKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case GAU:
						sum += covGauKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case SPH:
						sum += covSphKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + mean;	
}


__global__ void krigingSimpleAnisKernel_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, double alpha, double afac1, int numSrc, int nx, int ny, double mean)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	double sum=0.0f;
	double yr_x, yr_y;
	
	__shared__ double qs[BLOCK_SIZE_KRIGE1];
	__shared__ double Xs[BLOCK_SIZE_KRIGE1][2];

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
						sum += covExpAnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case GAU:
						sum += covGauAnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_2d(yr_x,yr_y,Xs[i][0],Xs[i][1],sill,range,nugget,alpha,afac1) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + mean;	
}

// Postprocessing conditional simulation

__global__ void addResSim_2d(double *res, double *uncond, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id];
}

__global__ void addResSimMean_2d(double *res, double *uncond, int n, double mean) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id] + mean;
}

//Funktionen fuer approximiertes Kriging

__global__ void overlay_2d(double2 *out, double2 *xy, double grid_minx, double grid_dx, double grid_maxy, double grid_dy, int numPoints) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    //Return als Fliesskommazahl fuer bilineare Interpolation
	if (i < numPoints) {
		out[i].x = (xy[i].x - grid_minx)/grid_dx;
		out[i].y = (grid_maxy - xy[i].y)/grid_dy;
	}
}


//fuer jede Realisierung
__global__ void residualsOrdinary_2d(double* res, double *srcdata, double *uncond_grid, double2 *indices, int nx, int ny, int numPoints) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numPoints) {
		
		// Bilinear interpolation
		double x = indices[id].x; 
		double y = indices[id].y;
		int row = floor(y); // y index of upper neighbour pixel
		int col = floor(x); // x index of lower neighbour pixel
		x = (double)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (double)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour

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
__global__ void residualsSimple_2d(double* res, double *srcdata, double *uncond_grid, double2 *indices, int nx, int ny, int numPoints, double mu) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numPoints) {
		
		// Bilinear interpolation
		double x = indices[id].x; 
		double y = indices[id].y;
		int row = floor(y); // y index of upper neighbour pixel
		int col = floor(x); // x index of lower neighbour pixel
		x = (double)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (double)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour

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
struct uncond_state_2d {
	cufftDoubleComplex *d_cov; // d_cov is the result of the preprocessing ans is needed for each realozation
	int nx,ny,n,m;
	double xmin,xmax,ymin,ymax,dx,dy;
	int blockSize,numBlocks;
	dim3 blockSize2, numBlocks2;
	cufftHandle plan1;
	dim3 blockSize2d;
	dim3 blockCount2d;
	dim3 blockSize1d;
	dim3 blockCount1d;
} uncond_global_2d;


#ifdef __cplusplus
extern "C" {
#endif


void EXPORT unconditionalSimInit_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, 
									double *p_sill, double *p_range, double *p_nugget, int *p_covmodel, double *p_anis_direction, 
									double *p_anis_ratio, int *do_check, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	
	uncond_global_2d.nx= *p_nx; // Number of cols
	uncond_global_2d.ny= *p_ny; // Number of rows
     
	//Grid wird einfach verdoppelt, nicht auf naechst 2er-Potenz erweitert
	uncond_global_2d.n= 2*uncond_global_2d.nx; // Number of cols
	uncond_global_2d.m= 2*uncond_global_2d.ny; // Number of rows
	//uncond_global_2d.n = ceil2(2*uncond_global_2d.nx); /// 
	//uncond_global_2d.m = ceil2(2*uncond_global_2d.ny); /// 
	uncond_global_2d.dx = (*p_xmax - *p_xmin) / (uncond_global_2d.nx-1);
	uncond_global_2d.dy = (*p_ymax - *p_ymin) / (uncond_global_2d.ny-1);
	
	// 1d cuda grid
	uncond_global_2d.blockSize1d = dim3(256);
	uncond_global_2d.blockCount1d = dim3(uncond_global_2d.n*uncond_global_2d.m / uncond_global_2d.blockSize1d.x);
	if (uncond_global_2d.n * uncond_global_2d.m % uncond_global_2d.blockSize1d.x  != 0) ++uncond_global_2d.blockCount1d.x;
	
	// 2d cuda grid
	uncond_global_2d.blockSize2d = dim3(16,16);
	uncond_global_2d.blockCount2d = dim3(uncond_global_2d.n / uncond_global_2d.blockSize2d.x, uncond_global_2d.m / uncond_global_2d.blockSize2d.y);
	if (uncond_global_2d.n % uncond_global_2d.blockSize2d.x != 0) ++uncond_global_2d.blockCount2d.x;
	if (uncond_global_2d.m % uncond_global_2d.blockSize2d.y != 0) ++uncond_global_2d.blockCount2d.y;

	
	//cufftPlan2d(&uncond_global_2d.plan1, uncond_global_2d.n, uncond_global_2d.m, CUFFT_Z2Z); 
	cufftPlan2d(&uncond_global_2d.plan1, uncond_global_2d.m, uncond_global_2d.n, CUFFT_Z2Z); 

	
	// build grid (ROW MAJOR)
	cufftDoubleComplex *h_grid_c = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.m*uncond_global_2d.n);
	for (int i=0; i<uncond_global_2d.n; ++i) { // i =  col index
		for (int j=0; j<uncond_global_2d.m; ++j) { // j = row index 
			h_grid_c[j*uncond_global_2d.n+i].x = *p_xmin + i * uncond_global_2d.dx; 
			//h_grid_c[j*uncond_global_2d.n+i].y = *p_ymin + (j+1) * uncond_global_2d.dy;  
			h_grid_c[j*uncond_global_2d.n+i].y = *p_ymin + (uncond_global_2d.m-1-j)* uncond_global_2d.dy; 
			//h_grid_c[j*uncond_global_2d.n+i].y = *p_ymax - j* uncond_global_2d.dy;

		}
	}
	
	
	double xc = *p_xmin + (uncond_global_2d.dx*uncond_global_2d.n)/2;
	double yc = *p_ymin +(uncond_global_2d.dy*uncond_global_2d.m)/2;
	double sill = *p_sill;
	double range = *p_range;
	double nugget = *p_nugget;
	bool isotropic = (*p_anis_ratio == 1.0);
	double afac1 = 1.0/(*p_anis_ratio);
	double alpha = (90.0 - *p_anis_direction) * (PI / 180.0);
	cufftDoubleComplex *d_grid;
	
	// Array for grid
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
	// Array for cov grid
	cudaStatus = cudaMalloc((void**)&uncond_global_2d.d_cov,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);

	// Sample covariance and generate "trick" grid
	cufftDoubleComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
	
	// copy grid to GPU
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, uncond_global_2d.n*uncond_global_2d.m*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice);
	
	if (isotropic) {
		sampleCovKernel_2d<<<uncond_global_2d.blockCount2d, uncond_global_2d.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global_2d.d_cov, xc, yc,*p_covmodel, sill, range,nugget,uncond_global_2d.n,uncond_global_2d.m);
	}
	else {	
		sampleCovAnisKernel_2d<<<uncond_global_2d.blockCount2d, uncond_global_2d.blockSize2d>>>(d_trick_grid_c, d_grid, uncond_global_2d.d_cov, xc, yc, *p_covmodel, sill, range,nugget, alpha, afac1, uncond_global_2d.n,uncond_global_2d.m);	
	}
	free(h_grid_c);
	cudaFree(d_grid);




	 
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE COV MATRIX******* ///////
//		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_2d.d_cov,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\sampleCov.csv",h_cov,uncond_global_2d.m,uncond_global_2d.n);
//		free(h_cov);
//	}
//#endif

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE TRICK GRID ******* /////// 
//		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\trickgrid.csv",h_cov,uncond_global_2d.m,uncond_global_2d.n);
//		free(h_cov);
//	}
//#endif
//


	// Execute 2d FFT of covariance grid in order to get the spectral representation 
	cufftExecZ2Z(uncond_global_2d.plan1, uncond_global_2d.d_cov, uncond_global_2d.d_cov, CUFFT_FORWARD); // in place fft forward


//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( COV GRID) ******* /////// 
//		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_2d.d_cov,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftSampleCov.csv",h_cov,uncond_global_2d.m,uncond_global_2d.n);
//		free(h_cov);
//	}
//#endif
//
	cufftExecZ2Z(uncond_global_2d.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forward

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( TRICK GRID) ******* /////// 
//		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftTrickGrid.csv",h_cov,uncond_global_2d.m,uncond_global_2d.n);
//		free(h_cov);
//	}
//#endif
	
	// Multiply fft of "trick" grid with n*m
	multKernel_2d<<<uncond_global_2d.blockCount1d, uncond_global_2d.blockSize1d>>>(d_trick_grid_c, uncond_global_2d.n, uncond_global_2d.m);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel_2d!\n", cudaStatus);	

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( TRICK GRID) ******* /////// 
//		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftTrickGridTimesNM.csv",h_cov,uncond_global_2d.m,uncond_global_2d.n);
//		free(h_cov);
//	}
//#endif


	// Devide spectral covariance grid by "trick" grid
	divideSpectrumKernel_2d<<<uncond_global_2d.blockCount1d, uncond_global_2d.blockSize1d>>>(uncond_global_2d.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel_f!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( COV GRID) / FFT(TRICKGRID)*N*M ******* /////// 
//		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_2d.d_cov,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftSampleCovByTrickGridNM.csv",h_cov,uncond_global_2d.m,uncond_global_2d.n);
//		free(h_cov);
//	}
//#endif
//


	// Copy to host and check for negative real parts
	if (*do_check) {
		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
		cudaStatus = cudaMemcpy(h_cov,uncond_global_2d.d_cov,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m,cudaMemcpyDeviceToHost);
		for (int i=0; i<uncond_global_2d.n*uncond_global_2d.m; ++i) {
			if (h_cov[i].x < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				free(h_cov);
				cudaFree(uncond_global_2d.d_cov);
				cufftDestroy(uncond_global_2d.plan1);
				return;
			}	
		}
		free(h_cov);
	}

	// Compute sqrt of cov grid
	sqrtKernel_2d<<<uncond_global_2d.blockCount1d, uncond_global_2d.blockSize1d>>>(uncond_global_2d.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel_f\n", cudaStatus);

}

// Generates unconditional realizations
// p_out = output array of size nx*ny*k * sizeof(double)
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT unconditionalSimRealizations_2d(double *p_out,  int *p_k, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus;

	int k = *p_k;

	double *d_rand; // device random numbers
	curandGenerator_t curandGen;
	cufftDoubleComplex *d_fftrand;
	cufftDoubleComplex* d_amp;
	double* d_out;

	curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGen,(unsigned long long)(time(NULL)));	
    
    //Realisierungen in Schleife, d.h. lineare Laufzeit in Bezug auf Realisierungen
	for(int l = 0; l<k; ++l) {
		
		if(l==0){			
			cudaStatus = cudaMalloc((void**)&d_rand,sizeof(double)*uncond_global_2d.m*uncond_global_2d.n); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
			// Generate Random Numbers
		curandGenerateNormalDouble(curandGen,d_rand,uncond_global_2d.m*uncond_global_2d.n,0.0,1.0);
		if(l==0) {
			cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftDoubleComplex) * uncond_global_2d.n * uncond_global_2d.m); 
			if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		}
		// Convert real random numbers to complex numbers
		realToComplexKernel_2d<<< uncond_global_2d.blockCount1d, uncond_global_2d.blockSize1d>>>(d_fftrand, d_rand, uncond_global_2d.n*uncond_global_2d.m);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel_f!\n", cudaStatus);	

		// Compute 2D FFT of random numbers
		cufftExecZ2Z(uncond_global_2d.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward

		if(l==0) cudaMalloc((void**)&d_amp,sizeof(cufftDoubleComplex)*uncond_global_2d.n*uncond_global_2d.m);
		elementProduct_2d<<<uncond_global_2d.blockCount1d, uncond_global_2d.blockSize1d>>>(d_amp, uncond_global_2d.d_cov, d_fftrand, uncond_global_2d.m*uncond_global_2d.n);  
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct_f!\n", cudaStatus);

		cufftExecZ2Z(uncond_global_2d.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation		
		if(l==0) cudaMalloc((void**)&d_out,sizeof(double)*uncond_global_2d.nx*uncond_global_2d.ny);
		
		dim3 blockSize2dhalf  = dim3(16,16);
		dim3 blockCount2dhalf = dim3(uncond_global_2d.nx/blockSize2dhalf.x,uncond_global_2d.ny/blockSize2dhalf.y);
		if (uncond_global_2d.nx % blockSize2dhalf.x != 0) ++blockCount2dhalf.x;
		if (uncond_global_2d.ny % blockSize2dhalf.y != 0) ++blockCount2dhalf.y;
		ReDiv_2d<<<blockCount2dhalf, blockSize2dhalf>>>(d_out, d_amp, std::sqrt((double)(uncond_global_2d.n*uncond_global_2d.m)), uncond_global_2d.nx, uncond_global_2d.ny, uncond_global_2d.n);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess) {
			printf("cudaThreadSynchronize returned error code %d after launching ReDiv_2d!\n", cudaStatus);
		}
		cudaMemcpy((p_out + l*(uncond_global_2d.nx*uncond_global_2d.ny)),d_out,sizeof(double)*uncond_global_2d.nx*uncond_global_2d.ny,cudaMemcpyDeviceToHost);
	}

	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_out);
	curandDestroyGenerator(curandGen);
}


void EXPORT unconditionalSimRelease_2d(int *ret_code) {
	*ret_code = OK;
	cudaFree(uncond_global_2d.d_cov);
	cufftDestroy(uncond_global_2d.plan1);
}


#ifdef __cplusplus
}
#endif
















/*******************************************************************************************
** CONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/


// global variables for conditional simulation that are needed both, for initialization as well as for generating realizations
struct cond_state_2d {
	cufftDoubleComplex *d_cov; 
	int nx,ny,n,m;
	double xmin,xmax,ymin,ymax,dx,dy;
	double range, sill, nugget;
	double alpha, afac1;
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
	double2 *d_samplexy; // coordinates of samples
	double2 *d_sampleindices; // Corresponding grid indices in subpixel accuracy
	double *d_sampledata; // data values of samples
	//double *d_covinv; // inverse covariance matrix of samples
	double *d_uncond;
	double *h_uncond; //cpu uncond cache
	bool uncond_gpucache;
	int covmodel;
	int k;
	double mu; // known mean for simple kriging
	int krige_method;
} cond_global_2d;













#ifdef __cplusplus
extern "C" {
#endif



//int *uncond_gpucache: wenn true, dann wird auf der GPU gespeichert (alte Implementierung), wenn false, dann Schleife

void EXPORT conditionalSimInit_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, 
								  int *p_ny, double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, 
								  double *p_srcData, int *p_numSrc, int *p_covmodel, double *p_anis_direction, 
								  double *p_anis_ratio, int *do_check, int *krige_method, double *mu, int *uncond_gpucache, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	cublasInit();

	cond_global_2d.nx= *p_nx; // Number of cols
	cond_global_2d.ny= *p_ny; // Number of rows
	cond_global_2d.n= 2*cond_global_2d.nx; // Number of cols
	cond_global_2d.m= 2*cond_global_2d.ny; // Number of rows
	cond_global_2d.dx = (*p_xmax - *p_xmin) / (cond_global_2d.nx - 1);
	cond_global_2d.dy = (*p_ymax - *p_ymin) / (cond_global_2d.ny - 1);
	cond_global_2d.numSrc = *p_numSrc;
	cond_global_2d.xmin = *p_xmin;
	cond_global_2d.xmax = *p_xmax;
	cond_global_2d.ymin = *p_ymin;
	cond_global_2d.ymax = *p_ymax;
	cond_global_2d.range = *p_range;
	cond_global_2d.sill = *p_sill;
	cond_global_2d.nugget = *p_nugget;
	cond_global_2d.covmodel = *p_covmodel;
	cond_global_2d.krige_method = *krige_method;
	if (cond_global_2d.krige_method == SIMPLE)
		cond_global_2d.mu = *mu;
	else cond_global_2d.mu = 0;
	cond_global_2d.isotropic = (*p_anis_ratio == 1.0);
	cond_global_2d.afac1 = 1.0/(*p_anis_ratio);
	cond_global_2d.alpha = (90.0 - *p_anis_direction) * (PI / 180.0);

	cond_global_2d.uncond_gpucache = (*uncond_gpucache != 0);

	// 1d cuda grid
	cond_global_2d.blockSize1d = dim3(256);
	cond_global_2d.blockCount1d = dim3(cond_global_2d.n*cond_global_2d.m / cond_global_2d.blockSize1d.x);
	if (cond_global_2d.n * cond_global_2d.m % cond_global_2d.blockSize1d.x  != 0) ++cond_global_2d.blockCount1d.x;
	
	// 2d cuda grid
	cond_global_2d.blockSize2d = dim3(16,16);
	cond_global_2d.blockCount2d = dim3(cond_global_2d.n / cond_global_2d.blockSize2d.x, cond_global_2d.m / cond_global_2d.blockSize2d.y);
	if (cond_global_2d.n % cond_global_2d.blockSize2d.x != 0) ++cond_global_2d.blockCount2d.x;
	if (cond_global_2d.m % cond_global_2d.blockSize2d.y != 0) ++cond_global_2d.blockCount2d.y;

	// 1d cuda grid for samples
	cond_global_2d.blockSizeSamples = dim3(256);
	cond_global_2d.blockCountSamples = dim3(cond_global_2d.numSrc / cond_global_2d.blockSizeSamples.x);
	if (cond_global_2d.numSrc % cond_global_2d.blockSizeSamples.x !=0) ++cond_global_2d.blockCountSamples.x;

	// Setup fft
	//cufftPlan2d(&cond_global_2d.plan1, cond_global_2d.n, cond_global_2d.m, CUFFT_Z2Z); // n und m vertauscht weil col major grid
	cufftPlan2d(&cond_global_2d.plan1, cond_global_2d.m, cond_global_2d.n, CUFFT_Z2Z); // n und m vertauscht weil col major grid

	
	
	// Build grid (ROW MAJOR)
	cufftDoubleComplex *h_grid_c = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*cond_global_2d.m*cond_global_2d.n);
	for (int i=0; i<cond_global_2d.n; ++i) { // i = col index
		for (int j=0; j<cond_global_2d.m; ++j) { // j = row index
			h_grid_c[j*cond_global_2d.n+i].x = *p_xmin + i * cond_global_2d.dx; 
			//h_grid_c[j*cond_global_2d.n+i].y = *p_ymin + (j+1) * cond_global_2d.dy;  
			h_grid_c[j*cond_global_2d.n+i].y = *p_ymin + (cond_global_2d.m-1-j)* cond_global_2d.dy;
			//h_grid_c[j*cond_global_2d.n+i].y = *p_ymax - j* cond_global_2d.dy;
		}
	}

	
	double xc = *p_xmin + (cond_global_2d.dx*cond_global_2d.n)/2;
	double yc = *p_ymin + (cond_global_2d.dy*cond_global_2d.m)/2;
	cufftDoubleComplex *d_grid;
	
	// Allocate grid and cov arrays on GPU
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(cufftDoubleComplex)*cond_global_2d.n*cond_global_2d.m);
	cudaStatus = cudaMalloc((void**)&cond_global_2d.d_cov,sizeof(cufftDoubleComplex)*cond_global_2d.n*cond_global_2d.m);

	// Sample covariance and generate "trick" grid
	cufftDoubleComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftDoubleComplex)*cond_global_2d.n*cond_global_2d.m);
	
	// Copy grid to gpu
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, cond_global_2d.n*cond_global_2d.m*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice);
	
	if (cond_global_2d.isotropic) {
		sampleCovKernel_2d<<<cond_global_2d.blockCount2d, cond_global_2d.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global_2d.d_cov, xc, yc, cond_global_2d.covmodel, cond_global_2d.sill, cond_global_2d.range, cond_global_2d.nugget, cond_global_2d.n,cond_global_2d.m);
	}
	else {
		sampleCovAnisKernel_2d<<<cond_global_2d.blockCount2d, cond_global_2d.blockSize2d>>>(d_trick_grid_c, d_grid, cond_global_2d.d_cov, xc, yc, cond_global_2d.covmodel, cond_global_2d.sill, cond_global_2d.range, cond_global_2d.nugget,cond_global_2d.alpha,cond_global_2d.afac1, cond_global_2d.n,cond_global_2d.m);
	}

	free(h_grid_c);
	cudaFree(d_grid);


	// Compute spectral representation of cov and "trick" grid
	cufftExecZ2Z(cond_global_2d.plan1, cond_global_2d.d_cov, cond_global_2d.d_cov, CUFFT_FORWARD); // in place fft forward
	cufftExecZ2Z(cond_global_2d.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forwar

	// Multiplication of fft(trick_grid) with n*m	
	multKernel_2d<<<cond_global_2d.blockCount1d, cond_global_2d.blockSize1d>>>(d_trick_grid_c, cond_global_2d.n, cond_global_2d.m);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel_2d!\n", cudaStatus);	

	// Devide spectral cov grid by fft of "trick" grid
	divideSpectrumKernel_2d<<<cond_global_2d.blockCount1d, cond_global_2d.blockSize1d>>>(cond_global_2d.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel_f!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	// Copy to host and check for negative real parts
	if (*do_check) {
		cufftDoubleComplex *h_cov = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*cond_global_2d.n*cond_global_2d.m);
		cudaStatus = cudaMemcpy(h_cov,cond_global_2d.d_cov,sizeof(cufftDoubleComplex)*cond_global_2d.n*cond_global_2d.m,cudaMemcpyDeviceToHost);
		for (int i=0; i<cond_global_2d.n*cond_global_2d.m; ++i) {
			if (h_cov[i].x < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				free(h_cov);
				cudaFree(cond_global_2d.d_cov);
				cufftDestroy(cond_global_2d.plan1);
				return;
			}	
		}
		free(h_cov);
	}

	// Compute sqrt of spectral cov grid
	sqrtKernel_2d<<<cond_global_2d.blockCount1d, cond_global_2d.blockSize1d>>>(cond_global_2d.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel_f\n", cudaStatus);

	// Copy samples to gpu
	cudaStatus = cudaMalloc((void**)&cond_global_2d.d_samplexy,sizeof(double2)* cond_global_2d.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global_2d.d_sampleindices,sizeof(double2)*cond_global_2d.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global_2d.d_sampledata,sizeof(double)*cond_global_2d.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMemcpy(cond_global_2d.d_samplexy,p_srcXY,sizeof(double2)* cond_global_2d.numSrc,cudaMemcpyHostToDevice);
	cudaMemcpy(cond_global_2d.d_sampledata,p_srcData,sizeof(double)*cond_global_2d.numSrc,cudaMemcpyHostToDevice);
		

	// Overlay samples to grid and save resulting subpixel grind indices
	overlay_2d<<<cond_global_2d.blockCountSamples,cond_global_2d.blockSizeSamples>>>(cond_global_2d.d_sampleindices,cond_global_2d.d_samplexy,*p_xmin,cond_global_2d.dx,*p_ymax,cond_global_2d.dy, cond_global_2d.numSrc);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay_2d!\n", cudaStatus);	
	// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access			

	/* TEST OUTPUT ON HOST */
	/*double2 *h_indices = (double2*)malloc(sizeof(double2)*cond_global_2d.numSrc);
	cudaMemcpy(h_indices,cond_global_2d.d_sampleindices,sizeof(double2)*cond_global_2d.numSrc,cudaMemcpyDeviceToHost);
	for (int i=0;i<cond_global_2d.numSrc;++i) {
		printf("(%.2f,%.2f) -> (%.2f,%.2f)\n",p_srcXY[2*i],p_srcXY[2*i+1],h_indices[i].x, h_indices[i].y);
	}
	free(h_indices);*/
}




// Generates Unconditional Realizations and the residuals of all samples to all realizations 
// p_out = output matrix of residuals, col means number of realization, row represents a sample point
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT conditionalSimUncondResiduals_2d(double *p_out, int *p_k, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	cond_global_2d.k = *p_k;
	
	double *d_rand; // Device Random Numbers
	curandGenerator_t curandGen;
	cufftDoubleComplex *d_fftrand;
	cufftDoubleComplex* d_amp;	
	double *d_residuals; // residuals of samples and underlying unconditional realization
	
	curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGen,(unsigned long long)(time(NULL)));	
	
	cudaStatus = cudaMalloc((void**)&d_rand,sizeof(double)*cond_global_2d.m*cond_global_2d.n); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftDoubleComplex) * cond_global_2d.n * cond_global_2d.m); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMalloc((void**)&d_amp,sizeof(cufftDoubleComplex)*cond_global_2d.n*cond_global_2d.m);
	cudaMalloc((void**)&cond_global_2d.d_uncond,sizeof(double)*cond_global_2d.nx*cond_global_2d.ny * cond_global_2d.k);

	/********************************************************
	GPU OR CPU CACHING OF UNCOND REALIZATIONS
    *********************************************************/

	if (cond_global_2d.uncond_gpucache) {
		cudaMalloc((void**)&cond_global_2d.d_uncond,sizeof(double)*cond_global_2d.nx*cond_global_2d.ny * cond_global_2d.k); // Hold all uncond realizations in GPU memory
	}
	else {
		cudaMalloc((void**)&cond_global_2d.d_uncond,sizeof(double)*cond_global_2d.nx*cond_global_2d.ny ); // Only one realization held in GPU memory
		cond_global_2d.h_uncond = (double*)malloc(sizeof(double)*cond_global_2d.nx*cond_global_2d.ny*cond_global_2d.k); // Hold all uncond realizations in CPU main memory
	}


	//Lagrange Multiplikatoren
	if (cond_global_2d.krige_method == ORDINARY) {
		cudaMalloc((void**)&d_residuals,sizeof(double)* (cond_global_2d.numSrc + 1));
	}
	else if (cond_global_2d.krige_method == SIMPLE) {
		cudaMalloc((void**)&d_residuals,sizeof(double)* cond_global_2d.numSrc);
	}
		
	for(int l=0; l<cond_global_2d.k; ++l) {
			
		
		curandGenerateNormalDouble(curandGen,d_rand,cond_global_2d.m*cond_global_2d.n,0.0,1.0);	
		realToComplexKernel_2d<<< cond_global_2d.blockCount1d, cond_global_2d.blockSize1d>>>(d_fftrand, d_rand, cond_global_2d.n*cond_global_2d.m);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel_f!\n", cudaStatus);	
		cufftExecZ2Z(cond_global_2d.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward
		cudaStatus = cudaThreadSynchronize();
		
		elementProduct_2d<<<cond_global_2d.blockCount1d, cond_global_2d.blockSize1d>>>(d_amp, cond_global_2d.d_cov, d_fftrand, cond_global_2d.m*cond_global_2d.n);
    
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct_f!\n", cudaStatus);

		cufftExecZ2Z(cond_global_2d.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation
	  
		dim3 blockSize2dhalf  = dim3(16,16);
		dim3 blockCount2dhalf = dim3(cond_global_2d.nx/blockSize2dhalf.x,cond_global_2d.ny/blockSize2dhalf.y);
		if (cond_global_2d.nx % blockSize2dhalf.x != 0) ++blockCount2dhalf.x;
		if (cond_global_2d.ny % blockSize2dhalf.y != 0) ++blockCount2dhalf.y;

        //GPU cache TRUE/FALSE
		
		if (cond_global_2d.uncond_gpucache) {
			ReDiv_2d<<<blockCount2dhalf, blockSize2dhalf>>>(cond_global_2d.d_uncond + l*cond_global_2d.nx*cond_global_2d.ny, d_amp, std::sqrt((double)(cond_global_2d.n*cond_global_2d.m)), cond_global_2d.nx, cond_global_2d.ny, cond_global_2d.n);
		}
		else {
			ReDiv_2d<<<blockCount2dhalf, blockSize2dhalf>>>(cond_global_2d.d_uncond, d_amp, std::sqrt((double)(cond_global_2d.n*cond_global_2d.m)), cond_global_2d.nx, cond_global_2d.ny, cond_global_2d.n);
		}
		
        //Brauchen wir diese Thread-Synchronisation? Wahrscheinlich nicht.
		cudaStatus = cudaThreadSynchronize();


		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching ReDiv_2d!\n", cudaStatus);
		
		// d_uncond is now a unconditional realization 
		// Compute residuals between samples and d_uncond
		if (cond_global_2d.uncond_gpucache) {
			if (cond_global_2d.krige_method == ORDINARY) {
				residualsOrdinary_2d<<<cond_global_2d.blockCountSamples,cond_global_2d.blockSizeSamples>>>(d_residuals,cond_global_2d.d_sampledata,cond_global_2d.d_uncond+l*(cond_global_2d.nx*cond_global_2d.ny),cond_global_2d.d_sampleindices,cond_global_2d.nx,cond_global_2d.ny,cond_global_2d.numSrc);
			}
			else if (cond_global_2d.krige_method == SIMPLE) {
				residualsSimple_2d<<<cond_global_2d.blockCountSamples,cond_global_2d.blockSizeSamples>>>(d_residuals,cond_global_2d.d_sampledata,cond_global_2d.d_uncond+l*(cond_global_2d.nx*cond_global_2d.ny),cond_global_2d.d_sampleindices,cond_global_2d.nx,cond_global_2d.ny,cond_global_2d.numSrc, cond_global_2d.mu);
			}
		}
		else {
			if (cond_global_2d.krige_method == ORDINARY) {
				residualsOrdinary_2d<<<cond_global_2d.blockCountSamples,cond_global_2d.blockSizeSamples>>>(d_residuals,cond_global_2d.d_sampledata,cond_global_2d.d_uncond,cond_global_2d.d_sampleindices,cond_global_2d.nx,cond_global_2d.ny,cond_global_2d.numSrc);
			}
			else if (cond_global_2d.krige_method == SIMPLE) {
				residualsSimple_2d<<<cond_global_2d.blockCountSamples,cond_global_2d.blockSizeSamples>>>(d_residuals,cond_global_2d.d_sampledata,cond_global_2d.d_uncond,cond_global_2d.d_sampleindices,cond_global_2d.nx,cond_global_2d.ny,cond_global_2d.numSrc, cond_global_2d.mu);
			}
		}

		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
	

		// Copy residuals to R, col major...
		if (cond_global_2d.krige_method == ORDINARY) {
			cudaMemcpy((p_out + l*(cond_global_2d.numSrc + 1)),d_residuals,sizeof(double)* (cond_global_2d.numSrc + 1),cudaMemcpyDeviceToHost);	
		}
		else if (cond_global_2d.krige_method == SIMPLE) {
			cudaMemcpy(p_out + l*cond_global_2d.numSrc,d_residuals,sizeof(double) * cond_global_2d.numSrc,cudaMemcpyDeviceToHost);	
		}

		// if needed, cache uncond realizations on cpu
		if (!cond_global_2d.uncond_gpucache) {
			cudaMemcpy(cond_global_2d.h_uncond+l*cond_global_2d.nx*cond_global_2d.ny,cond_global_2d.d_uncond, sizeof(double)*cond_global_2d.nx*cond_global_2d.ny,cudaMemcpyDeviceToHost);
		}
	}
	curandDestroyGenerator(curandGen);
	
	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_residuals);
}


void EXPORT conditionalSimKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus = cudaSuccess;
	
	double *d_y; // result vector from solving the kriging equation system
	double *d_respred; // interpolated residuals
	cudaMalloc((void**)&d_y, sizeof(double) * (cond_global_2d.numSrc + 1));
	cudaMalloc((void**)&d_respred, sizeof(double) * cond_global_2d.nx * cond_global_2d.ny);
	
	dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
	dim3 blockCntKrige = dim3((cond_global_2d.nx*cond_global_2d.ny) / blockSizeKrige.x);
	if ((cond_global_2d.nx*cond_global_2d.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;
	
	dim3 blockSizeCond = dim3(256);
	dim3 blockCntCond = dim3(cond_global_2d.nx*cond_global_2d.ny/ blockSizeCond.x);
	if (cond_global_2d.nx*cond_global_2d.ny % blockSizeCond.x != 0) ++blockCntCond.x; /// CHANGED!!!!

	for(int l = 0; l<cond_global_2d.k; ++l) {
						
		
		cudaMemcpy(d_y, p_y + l*(cond_global_2d.numSrc + 1), sizeof(double) * (cond_global_2d.numSrc + 1),cudaMemcpyHostToDevice);		
		
		// Kriging prediction		

		if (cond_global_2d.isotropic)
			krigingKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2d.d_samplexy,cond_global_2d.xmin,cond_global_2d.dx,cond_global_2d.ymin,cond_global_2d.dy,d_y,cond_global_2d.covmodel,cond_global_2d.range,cond_global_2d.sill,cond_global_2d.nugget,cond_global_2d.numSrc,cond_global_2d.nx,cond_global_2d.ny);
		else 	
			krigingAnisKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2d.d_samplexy,cond_global_2d.xmin,cond_global_2d.dx,cond_global_2d.ymin,cond_global_2d.dy,d_y,cond_global_2d.covmodel,cond_global_2d.range,cond_global_2d.sill,cond_global_2d.nugget,cond_global_2d.alpha,cond_global_2d.afac1,cond_global_2d.numSrc,cond_global_2d.nx,cond_global_2d.ny);


		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2d!\n", cudaStatus);
		
		// Add result to unconditional realization
		if (cond_global_2d.uncond_gpucache) {
			addResSim_2d<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2d.d_uncond + l*cond_global_2d.nx*cond_global_2d.ny, cond_global_2d.nx*cond_global_2d.ny);
		}
		else {
			cudaMemcpy(cond_global_2d.d_uncond, cond_global_2d.h_uncond+l*cond_global_2d.nx*cond_global_2d.ny, sizeof(double)*cond_global_2d.nx*cond_global_2d.ny,cudaMemcpyHostToDevice);
			addResSim_2d<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2d.d_uncond, cond_global_2d.nx*cond_global_2d.ny);	
		}
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_f!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global_2d.nx*cond_global_2d.ny)),d_respred,sizeof(double)*cond_global_2d.nx*cond_global_2d.ny,cudaMemcpyDeviceToHost);		
		
	}
	cudaFree(d_y);
	cudaFree(d_respred);
}




void EXPORT conditionalSimSimpleKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus = cudaSuccess;
	
	double *d_y; // result vector from solving the kriging equation system
	double *d_respred; // interpolated residuals
	cudaMalloc((void**)&d_y, sizeof(double) * cond_global_2d.numSrc); // not + 1, no lagrange multiplicator in simple kriging
	cudaMalloc((void**)&d_respred, sizeof(double) * cond_global_2d.nx * cond_global_2d.ny);
	
	dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
	dim3 blockCntKrige = dim3((cond_global_2d.nx*cond_global_2d.ny) / blockSizeKrige.x);
	if ((cond_global_2d.nx*cond_global_2d.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;
	
	dim3 blockSizeCond = dim3(256);
	dim3 blockCntCond = dim3(cond_global_2d.nx*cond_global_2d.ny/ blockSizeCond.x);
	if (cond_global_2d.nx*cond_global_2d.ny % blockSizeCond.x != 0) ++blockCntCond.x;

	for(int l = 0; l<cond_global_2d.k; ++l) {
						
		
		cudaMemcpy(d_y, p_y + l*cond_global_2d.numSrc, sizeof(double) * cond_global_2d.numSrc,cudaMemcpyHostToDevice);	// not + 1, no lagrange multiplicator in simple kriging		
		// Kriging prediction
		if (cond_global_2d.isotropic)
			krigingSimpleKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2d.d_samplexy,cond_global_2d.xmin,cond_global_2d.dx,cond_global_2d.ymin,cond_global_2d.dy,d_y,cond_global_2d.covmodel,cond_global_2d.range,cond_global_2d.sill,cond_global_2d.nugget,cond_global_2d.numSrc,cond_global_2d.nx,cond_global_2d.ny,cond_global_2d.mu);			
		else 	
			krigingSimpleAnisKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_2d.d_samplexy,cond_global_2d.xmin,cond_global_2d.dx,cond_global_2d.ymin,cond_global_2d.dy,d_y,cond_global_2d.covmodel,cond_global_2d.range,cond_global_2d.sill,cond_global_2d.nugget,cond_global_2d.alpha , cond_global_2d.afac1 ,cond_global_2d.numSrc,cond_global_2d.nx,cond_global_2d.ny,cond_global_2d.mu);			


		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2d!\n", cudaStatus);
		
		// Add result to unconditional realization
		if (cond_global_2d.uncond_gpucache) {
			addResSim_2d<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2d.d_uncond + l*cond_global_2d.nx*cond_global_2d.ny, cond_global_2d.nx*cond_global_2d.ny);
		}
		else {
			cudaMemcpy(cond_global_2d.d_uncond, cond_global_2d.h_uncond+l*cond_global_2d.nx*cond_global_2d.ny, sizeof(double)*cond_global_2d.nx*cond_global_2d.ny,cudaMemcpyHostToDevice);
			addResSim_2d<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_2d.d_uncond, cond_global_2d.nx*cond_global_2d.ny);	
		}

		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_f!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global_2d.nx*cond_global_2d.ny)),d_respred,sizeof(double)*cond_global_2d.nx*cond_global_2d.ny,cudaMemcpyDeviceToHost);		
		
	}
	cudaFree(d_y);
	cudaFree(d_respred);
}





void EXPORT conditionalSimRelease_2d(int *ret_code) {
	*ret_code = OK;
	cufftDestroy(cond_global_2d.plan1);
	cudaFree(cond_global_2d.d_samplexy);
	cudaFree(cond_global_2d.d_sampledata);
	cudaFree(cond_global_2d.d_sampleindices);
	cudaFree(cond_global_2d.d_cov);
	cudaFree(cond_global_2d.d_uncond);

	if (!cond_global_2d.uncond_gpucache) {
		free(cond_global_2d.h_uncond);
	}
}




#ifdef __cplusplus
}
#endif

























struct conditioning_state_2d {
	int nx,ny,n,m,k;
	double xmin,xmax,ymin,ymax,dx,dy;
	double range, sill, nugget;
	bool isotropic;
	double alpha,afac1;
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
	double2 *d_samplexy; 
	double2 *d_sampleindices; 
	double *d_sampledata; 
	//double *d_covinv; 
	double *d_uncond; // Unconditional realizations
	int covmodel;
	double mu; // known mean for simple kriging
	int krige_method;

} conditioning_global_2d;


#ifdef __cplusplus
extern "C" {
#endif


	void EXPORT conditioningInit_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, 
									double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, double *p_srcData, 
									int *p_numSrc,  int *p_k, double *p_uncond, int *p_covmodel, double *p_anis_direction, 
									double *p_anis_ratio, int *krige_method, double *mu, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus;
		cublasInit();

		conditioning_global_2d.nx= *p_nx; // Number of cols
		conditioning_global_2d.ny= *p_ny; // Number of rows
		conditioning_global_2d.n= 2*conditioning_global_2d.nx; // Number of cols
		conditioning_global_2d.m= 2*conditioning_global_2d.ny; // Number of rows
		conditioning_global_2d.dx = (*p_xmax - *p_xmin) / (conditioning_global_2d.nx-1);
		conditioning_global_2d.dy = (*p_ymax - *p_ymin) / (conditioning_global_2d.ny-1);
		conditioning_global_2d.numSrc = *p_numSrc;
		conditioning_global_2d.xmin = *p_xmin;
		conditioning_global_2d.xmax = *p_xmax;
		conditioning_global_2d.ymin = *p_ymin;
		conditioning_global_2d.ymax = *p_ymax;
		conditioning_global_2d.range = *p_range;
		conditioning_global_2d.sill = *p_sill;
		conditioning_global_2d.nugget = *p_nugget;
		conditioning_global_2d.k = *p_k;
		conditioning_global_2d.covmodel = *p_covmodel;
		conditioning_global_2d.krige_method = *krige_method;
		if (cond_global_2d.krige_method == SIMPLE)
			conditioning_global_2d.mu = *mu;
		else conditioning_global_2d.mu = 0;

		conditioning_global_2d.isotropic = (*p_anis_ratio == 1.0);
		conditioning_global_2d.afac1 = 1/(*p_anis_ratio);
		conditioning_global_2d.alpha = (90.0 - *p_anis_direction) * (PI / 180.0);

		// 1d cuda grid
		conditioning_global_2d.blockSize1d = dim3(256);
		conditioning_global_2d.blockCount1d = dim3(conditioning_global_2d.n*conditioning_global_2d.m / conditioning_global_2d.blockSize1d.x);
		if (conditioning_global_2d.n * conditioning_global_2d.m % conditioning_global_2d.blockSize1d.x  != 0) ++conditioning_global_2d.blockCount1d.x;
	
		// 2d cuda grid
		conditioning_global_2d.blockSize2d = dim3(16,16);
		conditioning_global_2d.blockCount2d = dim3(conditioning_global_2d.n / conditioning_global_2d.blockSize2d.x, conditioning_global_2d.m / conditioning_global_2d.blockSize2d.y);
		if (conditioning_global_2d.n % conditioning_global_2d.blockSize2d.x != 0) ++conditioning_global_2d.blockCount2d.x;
		if (conditioning_global_2d.m % conditioning_global_2d.blockSize2d.y != 0) ++conditioning_global_2d.blockCount2d.y;

		// 1d cuda grid der samples
		conditioning_global_2d.blockSizeSamples = dim3(256);
		conditioning_global_2d.blockCountSamples = dim3(conditioning_global_2d.numSrc / conditioning_global_2d.blockSizeSamples.x);
		if (conditioning_global_2d.numSrc % conditioning_global_2d.blockSizeSamples.x !=0) ++conditioning_global_2d.blockCountSamples.x;

		// Copy samples to gpu
		cudaStatus = cudaMalloc((void**)&conditioning_global_2d.d_samplexy,sizeof(double2)* conditioning_global_2d.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaStatus = cudaMalloc((void**)&conditioning_global_2d.d_sampleindices,sizeof(double2)*conditioning_global_2d.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaStatus = cudaMalloc((void**)&conditioning_global_2d.d_sampledata,sizeof(double)*conditioning_global_2d.numSrc); 
		if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
		cudaMemcpy(conditioning_global_2d.d_samplexy,p_srcXY,sizeof(double2)* conditioning_global_2d.numSrc,cudaMemcpyHostToDevice);
		cudaMemcpy(conditioning_global_2d.d_sampledata,p_srcData,sizeof(double)*conditioning_global_2d.numSrc,cudaMemcpyHostToDevice);
	

		
		// Overlay samples to grid and save resulting subpixel grid indices
		overlay_2d<<<conditioning_global_2d.blockCountSamples,conditioning_global_2d.blockSizeSamples>>>(conditioning_global_2d.d_sampleindices,conditioning_global_2d.d_samplexy,*p_xmin,conditioning_global_2d.dx,*p_ymax,conditioning_global_2d.dy, conditioning_global_2d.numSrc);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay_2d!\n", cudaStatus);	
		// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access	
		
		// Copy inverse sample cov matrix to gpu
		//cudaMalloc((void**)&conditioning_global_2d.d_covinv,sizeof(double) * (conditioning_global_2d.numSrc + 1) * (conditioning_global_2d.numSrc + 1));
		//cudaMemcpy(conditioning_global_2d.d_covinv,p_cov_inv,sizeof(double) * (conditioning_global_2d.numSrc + 1) * (conditioning_global_2d.numSrc + 1),cudaMemcpyHostToDevice);


		// Copy given unconditional realizations to gpu
		int size = sizeof(double) * conditioning_global_2d.nx * conditioning_global_2d.ny * conditioning_global_2d.k;
		cudaMalloc((void**)&conditioning_global_2d.d_uncond,size);
		cudaMemcpy(conditioning_global_2d.d_uncond,p_uncond,size,cudaMemcpyHostToDevice);
	}


	// Generates unconditional realizations
	// p_out = output array of size nx*ny*k * sizeof(double)
	// ret_code = return code: 0=ok
	void EXPORT conditioningResiduals_2d(double *p_out, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus;
			
		double *d_residuals; // residuals of sample data and unconditional realization
		
		if (conditioning_global_2d.krige_method == ORDINARY)
			cudaMalloc((void**)&d_residuals,sizeof(double)* (conditioning_global_2d.numSrc + 1));
		else if (conditioning_global_2d.krige_method == SIMPLE)
			cudaMalloc((void**)&d_residuals,sizeof(double)* (conditioning_global_2d.numSrc));

		for(int l = 0; l<conditioning_global_2d.k; ++l) {
			
			// d_uncond is now a unconditional realization 
			// Compute residuals between samples and d_uncond
			if (conditioning_global_2d.krige_method == ORDINARY) {
				residualsOrdinary_2d<<<conditioning_global_2d.blockCountSamples,conditioning_global_2d.blockSizeSamples>>>(d_residuals,conditioning_global_2d.d_sampledata,conditioning_global_2d.d_uncond+l*(conditioning_global_2d.nx*conditioning_global_2d.ny),conditioning_global_2d.d_sampleindices,conditioning_global_2d.nx,conditioning_global_2d.ny,conditioning_global_2d.numSrc);
			}
			else if (cond_global_2d.krige_method == SIMPLE) {
				residualsSimple_2d<<<conditioning_global_2d.blockCountSamples,conditioning_global_2d.blockSizeSamples>>>(d_residuals,conditioning_global_2d.d_sampledata,conditioning_global_2d.d_uncond+l*(conditioning_global_2d.nx*conditioning_global_2d.ny),conditioning_global_2d.d_sampleindices,conditioning_global_2d.nx,conditioning_global_2d.ny,conditioning_global_2d.numSrc,conditioning_global_2d.mu);
			}
			cudaStatus = cudaThreadSynchronize();	
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
	

			// Copy residuals to R, col major...
			if (conditioning_global_2d.krige_method == ORDINARY) {
				cudaMemcpy((p_out + l*(conditioning_global_2d.numSrc + 1)),d_residuals,sizeof(double)* (conditioning_global_2d.numSrc + 1),cudaMemcpyDeviceToHost);	
			}
			else if (conditioning_global_2d.krige_method == SIMPLE) {
				cudaMemcpy(p_out + l*conditioning_global_2d.numSrc,d_residuals,sizeof(double) * conditioning_global_2d.numSrc,cudaMemcpyDeviceToHost);	
			}
		}	
		cudaFree(d_residuals);
	}


	void EXPORT conditioningKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus = cudaSuccess;

		double *d_y; // result vector of kriging equation system
		double *d_respred; // Interpolated grid of residuals

		cudaMalloc((void**)&d_y, sizeof(double) * (conditioning_global_2d.numSrc + 1));
		cudaMalloc((void**)&d_respred, sizeof(double) * conditioning_global_2d.nx * conditioning_global_2d.ny);

		dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
		dim3 blockCntKrige = dim3((conditioning_global_2d.nx*conditioning_global_2d.ny) / blockSizeKrige.x);
		if ((conditioning_global_2d.nx*conditioning_global_2d.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;

		dim3 blockSizeCond = dim3(256);
		dim3 blockCntCond = dim3(conditioning_global_2d.nx*conditioning_global_2d.ny/ blockSizeCond.x);
		if (conditioning_global_2d.nx*conditioning_global_2d.ny % blockSizeCond.x != 0) ++blockCntCond.x;

		for(int l = 0; l<cond_global_2d.k; ++l) {
			cudaMemcpy(d_y, p_y + l*(conditioning_global_2d.numSrc + 1), sizeof(double) * (conditioning_global_2d.numSrc + 1),cudaMemcpyHostToDevice);		
					
			if (conditioning_global_2d.isotropic) {
				krigingKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2d.d_samplexy,conditioning_global_2d.xmin,conditioning_global_2d.dx,conditioning_global_2d.ymin,conditioning_global_2d.dy,d_y,conditioning_global_2d.covmodel,conditioning_global_2d.range,conditioning_global_2d.sill,conditioning_global_2d.nugget,conditioning_global_2d.numSrc,conditioning_global_2d.nx,conditioning_global_2d.ny);			
			}
			else {
				krigingAnisKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2d.d_samplexy,conditioning_global_2d.xmin,conditioning_global_2d.dx,conditioning_global_2d.ymin,conditioning_global_2d.dy,d_y,conditioning_global_2d.covmodel,conditioning_global_2d.range,conditioning_global_2d.sill,conditioning_global_2d.nugget,conditioning_global_2d.alpha,conditioning_global_2d.afac1,conditioning_global_2d.numSrc,conditioning_global_2d.nx,conditioning_global_2d.ny);			
			}
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2d!\n", cudaStatus);
		
			// Add result to unconditional realization	
			addResSim_2d<<<blockCntCond,blockSizeCond>>>(d_respred, &conditioning_global_2d.d_uncond[l*conditioning_global_2d.nx*conditioning_global_2d.ny], conditioning_global_2d.nx*conditioning_global_2d.ny);
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_f!\n", cudaStatus);
		
			// Write result to R
			cudaMemcpy((p_out + l*(conditioning_global_2d.nx*conditioning_global_2d.ny)),d_respred,sizeof(double)*conditioning_global_2d.nx*conditioning_global_2d.ny,cudaMemcpyDeviceToHost);		
		
		}

		cudaFree(d_y);
		cudaFree(d_respred);
	}


    //Nicht implementiert fuer 3D.

	void EXPORT conditioningSimpleKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code) {
		*ret_code = OK;
		cudaError_t cudaStatus = cudaSuccess;

		double *d_y; // result vector of kriging equation system
		double *d_respred; // Interpolated grid of residuals

		cudaMalloc((void**)&d_y, sizeof(double) * conditioning_global_2d.numSrc);
		cudaMalloc((void**)&d_respred, sizeof(double) * conditioning_global_2d.nx * conditioning_global_2d.ny);

		dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
		dim3 blockCntKrige = dim3((conditioning_global_2d.nx*conditioning_global_2d.ny) / blockSizeKrige.x);
		if ((conditioning_global_2d.nx*conditioning_global_2d.ny) % blockSizeKrige.x != 0) ++blockCntKrige.x;

		dim3 blockSizeCond = dim3(256);
		dim3 blockCntCond = dim3(conditioning_global_2d.nx*conditioning_global_2d.ny/ blockSizeCond.x);
		if (conditioning_global_2d.nx*conditioning_global_2d.ny % blockSizeCond.x != 0) ++blockCntCond.x;

		for(int l = 0; l<cond_global_2d.k; ++l) {
			cudaMemcpy(d_y, p_y + l*conditioning_global_2d.numSrc, sizeof(double) * conditioning_global_2d.numSrc,cudaMemcpyHostToDevice);		
					
			if (conditioning_global_2d.isotropic) {
				krigingSimpleKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2d.d_samplexy,conditioning_global_2d.xmin,conditioning_global_2d.dx,conditioning_global_2d.ymin,conditioning_global_2d.dy,d_y,conditioning_global_2d.covmodel,conditioning_global_2d.range,conditioning_global_2d.sill,conditioning_global_2d.nugget,conditioning_global_2d.numSrc,conditioning_global_2d.nx,conditioning_global_2d.ny,conditioning_global_2d.mu);
			}
			else {
				krigingSimpleAnisKernel_2d<<<blockCntKrige, blockSizeKrige>>>(d_respred,conditioning_global_2d.d_samplexy,conditioning_global_2d.xmin,conditioning_global_2d.dx,conditioning_global_2d.ymin,conditioning_global_2d.dy,d_y,conditioning_global_2d.covmodel,conditioning_global_2d.range,conditioning_global_2d.sill,conditioning_global_2d.nugget,conditioning_global_2d.alpha,conditioning_global_2d.afac1,conditioning_global_2d.numSrc,conditioning_global_2d.nx,conditioning_global_2d.ny,conditioning_global_2d.mu);
			}
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_2d!\n", cudaStatus);
		
			// Add result to unconditional realization	
			addResSim_2d<<<blockCntCond,blockSizeCond>>>(d_respred, &conditioning_global_2d.d_uncond[l*conditioning_global_2d.nx*conditioning_global_2d.ny], conditioning_global_2d.nx*conditioning_global_2d.ny);
			if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_f!\n", cudaStatus);
		
			// Write result to R
			cudaMemcpy((p_out + l*(conditioning_global_2d.nx*conditioning_global_2d.ny)),d_respred,sizeof(double)*conditioning_global_2d.nx*conditioning_global_2d.ny,cudaMemcpyDeviceToHost);		
		
		}

		cudaFree(d_y);
		cudaFree(d_respred);
	}



	void EXPORT conditioningRelease_2d(int *ret_code) {
		*ret_code = OK;
		cudaFree(conditioning_global_2d.d_uncond);
		cudaFree(conditioning_global_2d.d_samplexy);
		cudaFree(conditioning_global_2d.d_sampledata);
		cudaFree(conditioning_global_2d.d_sampleindices);
	}

#ifdef __cplusplus
}
#endif