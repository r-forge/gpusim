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

__device__ float covExpKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget) {
	float dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by)+(az-bz)*(az-bz));
	return ((dist == 0.0f)? (nugget + sill) : (sill*exp(-dist/range)));
}


__device__ float covExpAnisKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget, float alpha, float beta, float theta, float afac1, float afac2) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	float dz = az-bz;	
	temp = dx*cos(beta)*cos(alpha) + dy*cos(beta)*sin(alpha) - dz * sin(beta);
	dist += temp * temp;
	temp = afac1 * (-dx * (cos(theta)*sin(alpha) + sin(theta)*sin(beta)*cos(alpha)) + 
						dy * (cos(theta)*cos(alpha) + sin(theta)*sin(beta)*sin(alpha)) + 
						dz * sin(theta)*cos(beta));
	dist += temp * temp;
	temp = afac2 * (dx * (sin(theta)*sin(alpha) + cos(theta)*sin(beta)*cos(alpha)) + 
					dy * (-sin(theta)*cos(alpha) + cos(theta)*sin(beta)*sin(alpha)) + 
					dz * cos(theta) * cos(beta));		
	dist += temp * temp;
	dist = sqrt(dist);
	return ((dist == 0.0f)? (nugget + sill) : (sill*exp(-dist/range)));
}



__device__ float covGauKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget) {
	float dist2 = (ax-bx)*(ax-bx)+(ay-by)*(ay-by)+(az-bz)*(az-bz);
	return ((dist2 == 0.0f)? (nugget + sill) : (sill*exp(-dist2/(range*range))));
}



__device__ float covGauAnisKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget, float alpha, float beta, float theta, float afac1, float afac2) {
	
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	float dz = az-bz;	
	temp = dx*cos(beta)*cos(alpha) + dy*cos(beta)*sin(alpha) - dz * sin(beta);
	dist += temp * temp;
	temp = afac1 * (-dx * (cos(theta)*sin(alpha) + sin(theta)*sin(beta)*cos(alpha)) + 
						dy * (cos(theta)*cos(alpha) + sin(theta)*sin(beta)*sin(alpha)) + 
						dz * sin(theta)*cos(beta));
	dist += temp * temp;
	temp = afac2 * (dx * (sin(theta)*sin(alpha) + cos(theta)*sin(beta)*cos(alpha)) + 
					dy * (-sin(theta)*cos(alpha) + cos(theta)*sin(beta)*sin(alpha)) + 
					dz * cos(theta) * cos(beta));		
	dist += temp * temp;
	//dist = sqrt(dist);
	return ((dist == 0.0f)? (nugget + sill) : (sill*exp(-dist/(range*range))));
}





__device__ float covSphKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget) {
	float dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by)+(az-bz)*(az-bz));
	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0 - (((3.0*dist) / (2.0*range)) - ((dist * dist * dist) / (2.0 * range * range * range)) ));	
	return 0.0f; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}




__device__ float covSphAnisKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget, float alpha, float beta, float theta, float afac1, float afac2) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	float dz = az-bz;	
	temp = dx*cos(beta)*cos(alpha) + dy*cos(beta)*sin(alpha) - dz * sin(beta);
	dist += temp * temp;
	temp = afac1 * (-dx * (cos(theta)*sin(alpha) + sin(theta)*sin(beta)*cos(alpha)) + 
						dy * (cos(theta)*cos(alpha) + sin(theta)*sin(beta)*sin(alpha)) + 
						dz * sin(theta)*cos(beta));
	dist += temp * temp;
	temp = afac2 * (dx * (sin(theta)*sin(alpha) + cos(theta)*sin(beta)*cos(alpha)) + 
					dy * (-sin(theta)*cos(alpha) + cos(theta)*sin(beta)*sin(alpha)) + 
					dz * cos(theta) * cos(beta));		
	dist += temp * temp;
	dist = sqrt(dist);
	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0 - (((3.0*dist) / (2.0*range)) - ((dist * dist * dist) / (2.0 * range * range * range)) ));	
	return 0.0f; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}












__device__ float covMat3Kernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by)+(az-bz)*(az-bz));
	return ((dist == 0.0f)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*exp(-SQRT3*dist/range)));
}


__device__ float covMat3AnisKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget, float alpha, float beta, float theta, float afac1, float afac2) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	float dz = az-bz;	
	temp = dx*cos(beta)*cos(alpha) + dy*cos(beta)*sin(alpha) - dz * sin(beta);
	dist += temp * temp;
	temp = afac1 * (-dx * (cos(theta)*sin(alpha) + sin(theta)*sin(beta)*cos(alpha)) + 
						dy * (cos(theta)*cos(alpha) + sin(theta)*sin(beta)*sin(alpha)) + 
						dz * sin(theta)*cos(beta));
	dist += temp * temp;
	temp = afac2 * (dx * (sin(theta)*sin(alpha) + cos(theta)*sin(beta)*cos(alpha)) + 
					dy * (-sin(theta)*cos(alpha) + cos(theta)*sin(beta)*sin(alpha)) + 
					dz * cos(theta) * cos(beta));		
	dist += temp * temp;
	dist = sqrt(dist);
	return ((dist == 0.0f)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*exp(-SQRT3*dist/range)));
}




__device__ float covMat5Kernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget) {
	float dist = sqrtf((ax-bx)*(ax-bx)+(ay-by)*(ay-by)+(az-bz)*(az-bz));
	return ((dist == 0.0f)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * exp(-SQRT5*dist/range)));
}
__device__ float covMat5AnisKernel_3f(float ax, float ay, float az, float bx, float by, float bz, float sill, float range, float nugget, float alpha, float beta, float theta, float afac1, float afac2) {
	float dist = 0.0;
	float temp = 0.0;
	float dx = ax-bx;
	float dy = ay-by;
	float dz = az-bz;	
	temp = dx*cos(beta)*cos(alpha) + dy*cos(beta)*sin(alpha) - dz * sin(beta);
	dist += temp * temp;
	temp = afac1 * (-dx * (cos(theta)*sin(alpha) + sin(theta)*sin(beta)*cos(alpha)) + 
						dy * (cos(theta)*cos(alpha) + sin(theta)*sin(beta)*sin(alpha)) + 
						dz * sin(theta)*cos(beta));
	dist += temp * temp;
	temp = afac2 * (dx * (sin(theta)*sin(alpha) + cos(theta)*sin(beta)*cos(alpha)) + 
					dy * (-sin(theta)*cos(alpha) + cos(theta)*sin(beta)*sin(alpha)) + 
					dz * cos(theta) * cos(beta));		
	dist += temp * temp;
	dist = sqrt(dist);
	return ((dist == 0.0f)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * exp(-SQRT5*dist/range)));
}


// Converts real float array into cufftComplex array
__global__ void realToComplexKernel_3f(cufftComplex *c, float* r, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		c[i].x = r[i];
		c[i].y = 0.0f;
	}
}

// TODO: CHECK CORECTNESS
__global__ void ReDiv_3f(float *out, cufftComplex *c, float div,int nx, int ny, int nz) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int zdim = threadIdx.z + blockIdx.z * blockDim.z;
	if (col < nx && row < ny && zdim < nz) out[zdim*nx*ny + row*nx + col] = c[zdim*4*nx*ny +  row*2*nx + col].x / div; //////// !!!!!!
}




// Covariance sampling of a regular grid
__global__ void sampleCovKernel_3f(cufftComplex *trickgrid,float3 *grid, cufftComplex* cov, float xc, float yc, float zc, int model, float sill, float range, float nugget, int n, int m, int o) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int zdim = threadIdx.z + blockIdx.z * blockDim.z;
	if (col < n && row < m && zdim < o) {
		
		
		switch (model) {
		case EXP:
			cov[zdim*n*m + row*n + col].x = covExpKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget);
			break;
		case GAU:
			cov[zdim*n*m + row*n + col].x = covGauKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget);
			break;
		case SPH:
			cov[zdim*n*m + row*n + col].x = covSphKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget);
			break;
		case MAT3:
			cov[zdim*n*m + row*n + col].x = covMat3Kernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget);
			break;
		case MAT5:
			cov[zdim*n*m + row*n + col].x = covMat5Kernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget);
			break;
		}	
	

		cov[zdim*n*m + row*n + col].y = 0;	
		if (col == n/2-1 && row == m/2-1 && zdim == o/2-1) {
			trickgrid[zdim*n*m + row*n+col].x = 1.0f;
			trickgrid[zdim*n*m + row*n+col].y = 0.0f;
		}
		else {
			trickgrid[zdim*n*m + row*n+col].x = 0.0f;
			trickgrid[zdim*n*m + row*n+col].y = 0.0f;
		}
	}
}






__global__ void sampleCovAnisKernel_3f(cufftComplex *trickgrid,float3 *grid, cufftComplex* cov, float xc, float yc, float zc, int model, float sill, float range, float nugget, float alpha, float beta, float theta, float afac1, float afac2, int n, int m, int o) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int zdim = threadIdx.z + blockIdx.z * blockDim.z;
	if (col < n && row < m && zdim < o) {	
		switch (model) {
		case EXP:
			cov[zdim*n*m + row*n + col].x = covExpAnisKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget,alpha,beta,theta,afac1,afac2);
			break;
		case GAU:
			cov[zdim*n*m + row*n + col].x = covGauAnisKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget,alpha,beta,theta,afac1,afac2);
			break;
		case SPH:
			cov[zdim*n*m + row*n + col].x = covSphAnisKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget,alpha,beta,theta,afac1,afac2);
			break;
		case MAT3:
			cov[zdim*n*m + row*n + col].x = covMat3AnisKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget,alpha,beta,theta,afac1,afac2);
			break;
		case MAT5:
			cov[zdim*n*m + row*n + col].x = covMat5AnisKernel_3f(grid[zdim*n*m +row*n+col].x,grid[zdim*n*m + row*n+col].y,grid[zdim*n*m + row*n+col].z,xc,yc,zc,sill,range,nugget,alpha,beta,theta,afac1,afac2);
			break;
		}	
		cov[zdim*n*m + row*n + col].y = 0;	
		if (col == n/2-1 && row == m/2-1 && zdim == o/2-1) {
			trickgrid[zdim*n*m + row*n+col].x = 1.0f;
			trickgrid[zdim*n*m + row*n+col].y = 0.0f;
		}
		else {
			trickgrid[zdim*n*m + row*n+col].x = 0.0f;
			trickgrid[zdim*n*m + row*n+col].y = 0.0f;
		}
	}
}










// TODO: Calculate n*m*o before on cpu and use this value as one arg instead of 3!!!!
__global__ void multKernel_3f(cufftComplex *fftgrid, int n, int m, int o) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	fftgrid[i].x = fftgrid[i].x*n*m*o;
	fftgrid[i].y = fftgrid[i].y*n*m*o;
}


// Devides spectral grid elementwise by fftgrid
__global__ void divideSpectrumKernel_3f(cufftComplex *spectrum, cufftComplex *fftgrid) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float a = spectrum[i].x;
	float b = spectrum[i].y;
	float c = fftgrid[i].x;
	float d = fftgrid[i].y;
	spectrum[i].x = (a*c+b*d)/(c*c+d*d);
	spectrum[i].y = (b*c-a*d)/(c*c+d*d);
}



// Element-wise sqrt from spectral grid
__global__ void sqrtKernel_3f(cufftComplex *spectrum) {
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
__global__ void elementProduct_3f(cufftComplex *c, cufftComplex *a, cufftComplex *b, int n) {
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



// TODO: KRIGING -> 3d

__global__ void addResSim_3f(float *res, float *uncond, int n) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id];
}

__global__ void addResSimMean_3f(float *res, float *uncond, int n, float mean) 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) res[id] += uncond[id] + mean;
}


__global__ void overlay_3f(float3 *out, float3 *xy, float grid_minx, float grid_dx, float grid_maxy, float grid_dy, float grid_minz, float grid_dz, int numPoints) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPoints) {
		out[i].x = (xy[i].x - grid_minx)/grid_dx;
		out[i].y = (grid_maxy - grid_dy - xy[i].y)/grid_dy;
		out[i].z = (xy[i].z - grid_minz)/grid_dz;
	}
}


__global__ void residualsOrdinary_3f(float* res, float *srcdata, float *uncond_grid, float3 *indices, int nx, int ny, int nz, int numPoints) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numPoints) {
		
		// Trilinear interpolation
		float x = indices[id].x; 
		float y = indices[id].y;
		float z = indices[id].z;
		int row = floor(y); // y index of upper neighbour pixel
		int col = floor(x); // x index of lower neighbour pixel
		int zdim = floor(z); // z index of lower neighbour pixel
		x = (float)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (float)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour
		z = (float)z - zdim;
		
		// Special cases
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
		if (zdim > nz-1) {
			z = 0.0f;zdim = nz-1;
		}
		else if (zdim < 0) {
			z = 0.0f;zdim = 0;
		}

		float c00 = (1-y)*uncond_grid[zdim*nx*ny + row*nx + col]         +   y*uncond_grid[zdim*nx*ny + row*nx + col+1];
		float c10 = (1-y)*uncond_grid[(zdim+1)*nx*ny + row*nx + col]     +   y*uncond_grid[(zdim+1)*nx*ny + row*nx + col+1];
		float c01 = (1-y)*uncond_grid[zdim*nx*ny + (row+1)*nx + col]     +   y*uncond_grid[zdim*nx*ny + (row+1)*nx + col+1];
		float c11 = (1-y)*uncond_grid[(zdim+1)*nx*ny + (row+1)*nx + col] +   y*uncond_grid[(zdim+1)*nx*ny + (row+1)*nx + col+1];

		c00 = (1-z) * c00 + z*c10;
		c01 = (1-z) * c01 + z*c11;

		res[id] = srcdata[id] - ((1-x)*c00 + x*c01);
		
		/*res[id] = srcdata[id] - ((1-y) * ((1-x) * uncond_grid[row * nx + col] + x * uncond_grid[row * nx + col + 1]) + 
								  y * ((1-x) * uncond_grid[(row+1) * nx + col] + x * uncond_grid[(row+1) * nx + col + 1]));*/
	}		
	if (id == 0) {
		res[numPoints] = 0.0f; // Needed as Lagrange factor for GEMV with inverse covariance matrix of samples (needed for Kriging)
	}
}



// Calculates residuals of samples and an unconditional realization. Uses bilinear interpolation based on the sample's position in grid
__global__ void residualsSimple_3f(float* res, float *srcdata, float *uncond_grid, float3 *indices, int nx, int ny, int nz, int numPoints, float mu) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numPoints) {
		
		// Trilinear interpolation
		float x = indices[id].x; 
		float y = indices[id].y;
		float z = indices[id].z;
		int row = floor(y); // y index of upper neighbour pixel
		int col = floor(x); // x index of lower neighbour pixel
		int zdim = floor(z); // z index of lower neighbour pixel
		x = (float)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (float)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour
		z = (float)z - zdim;
		
		// Special cases
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
		if (zdim > nz-1) {
			z = 0.0f;zdim = nz-1;
		}
		else if (zdim < 0) {
			z = 0.0f;zdim = 0;
		}

		float c00 = (1-y)*uncond_grid[zdim*nx*ny + row*nx + col]         +   y*uncond_grid[zdim*nx*ny + row*nx + col+1];
		float c10 = (1-y)*uncond_grid[(zdim+1)*nx*ny + row*nx + col]     +   y*uncond_grid[(zdim+1)*nx*ny + row*nx + col+1];
		float c01 = (1-y)*uncond_grid[zdim*nx*ny + (row+1)*nx + col]     +   y*uncond_grid[zdim*nx*ny + (row+1)*nx + col+1];
		float c11 = (1-y)*uncond_grid[(zdim+1)*nx*ny + (row+1)*nx + col] +   y*uncond_grid[(zdim+1)*nx*ny + (row+1)*nx + col+1];

		c00 = (1-z) * c00 + z*c10;
		c01 = (1-z) * c01 + z*c11;

		res[id] = srcdata[id] - ((1-x)*c00 + x*c01);
		
		/*res[id] = srcdata[id] - ((1-y) * ((1-x) * uncond_grid[row * nx + col] + x * uncond_grid[row * nx + col + 1]) + 
								  y * ((1-x) * uncond_grid[(row+1) * nx + col] + x * uncond_grid[(row+1) * nx + col + 1]));*/
	}		
}








/// Kriging prediction at a regular grid with given samples for conditioning
#ifndef BLOCK_SIZE_KRIGE1
#define BLOCK_SIZE_KRIGE1 256
#endif


__global__ void krigingKernel_3f(float *prediction, float3 *srcXY, float xmin, float dx, float ymin, float dy, float zmin, float dz, float *y, int model, float range, float sill, float nugget, int numSrc, int nx, int ny, int nz)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0;
	float yr_x, yr_y, yr_z;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][3];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){
		yr_z = zmin + dz * (int)((bx*BLOCK_SIZE_KRIGE1 + tx)/(nx*ny));
		yr_y = ymin + dy * (ny-1-   (int)(((bx*BLOCK_SIZE_KRIGE1 + tx)%(nx*ny))/nx));
		yr_x = xmin + dx * ((bx*BLOCK_SIZE_KRIGE1 + tx)%nx);

	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			Xs[tx][2]=srcXY[(tx+b)].z;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					switch (model) {
					case EXP:
						sum += covExpKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i];  
						break;
					case GAU:
						sum += covGauKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					case SPH:
						sum += covSphKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}





__global__ void krigingAnisKernel_3f(float *prediction, float3 *srcXY, float xmin, float dx, float ymin, float dy, float zmin, float dz, float *y, int model, float range, float sill, float nugget, float alpha, float beta, float theta, float afac1, float afac2, int numSrc, int nx, int ny, int nz)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0;
	float yr_x, yr_y, yr_z;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][3];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){
		yr_z = zmin + dz * (int)((bx*BLOCK_SIZE_KRIGE1 + tx)/(nx*ny));
		yr_y = ymin + dy * (ny-1-   (int)(((bx*BLOCK_SIZE_KRIGE1 + tx)%(nx*ny))/nx));
		yr_x = xmin + dx * ((bx*BLOCK_SIZE_KRIGE1 + tx)%nx);

	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			Xs[tx][2]=srcXY[(tx+b)].z;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					switch (model) {
					case EXP:
						sum += covExpAnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i];  
						break;
					case GAU:
						sum += covGauAnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + y[numSrc];	
}






__global__ void krigingSimpleKernel_3f(float *prediction, float3 *srcXY, float xmin, float dx, float ymin, float dy, float zmin, float dz, float *y, int model, float range, float sill, float nugget, int numSrc, int nx, int ny, int nz, float mean)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0;
	float yr_x, yr_y, yr_z;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][3];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){
		yr_z = zmin + dz * (int)((bx*BLOCK_SIZE_KRIGE1 + tx)/(nx*ny));
		yr_y = ymin + dy * (ny-1-   (int)(((bx*BLOCK_SIZE_KRIGE1 + tx)%(nx*ny))/nx));
		yr_x = xmin + dx * ((bx*BLOCK_SIZE_KRIGE1 + tx)%nx);

	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			Xs[tx][2]=srcXY[(tx+b)].z;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					switch (model) {
					case EXP:
						sum += covExpKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i];  
						break;
					case GAU:
						sum += covGauKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					case SPH:
						sum += covSphKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + mean;	
}






__global__ void krigingSimpleAnisKernel_3f(float *prediction, float3 *srcXY, float xmin, float dx, float ymin, float dy, float zmin, float dz, float *y, int model, float range, float sill, float nugget, float alpha, float beta, float theta, float afac1, float afac2, int numSrc, int nx, int ny, int nz, float mean)
{	
	int bx = blockIdx.x;
    int tx = threadIdx.x;

	float sum=0.0;
	float yr_x, yr_y, yr_z;
	
	__shared__ float qs[BLOCK_SIZE_KRIGE1];
	__shared__ float Xs[BLOCK_SIZE_KRIGE1][3];

    if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){
		yr_z = zmin + dz * (int)((bx*BLOCK_SIZE_KRIGE1 + tx)/(nx*ny));
		yr_y = ymin + dy * (ny-1-   (int)(((bx*BLOCK_SIZE_KRIGE1 + tx)%(nx*ny))/nx));
		yr_x = xmin + dx * ((bx*BLOCK_SIZE_KRIGE1 + tx)%nx);

	}
	__syncthreads();
	for (int b=0;b<numSrc;b+=BLOCK_SIZE_KRIGE1){
		
		if ((b+tx)<numSrc){         
			Xs[tx][0]=srcXY[(tx+b)].x;
			Xs[tx][1]=srcXY[(tx+b)].y;
			Xs[tx][2]=srcXY[(tx+b)].z;
			qs[tx]=y[tx+b];
		}
		__syncthreads();
		if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz){	
			for (int i=0;i<BLOCK_SIZE_KRIGE1;++i){
				if ((b+i)<numSrc){
					switch (model) {
					case EXP:
						sum += covExpAnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i];  
						break;
					case GAU:
						sum += covGauAnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_3f(yr_x,yr_y,yr_z,Xs[i][0],Xs[i][1],Xs[i][2],sill,range,nugget,alpha,beta,theta,afac1,afac2) *qs[i]; 
						break;
					}
				}
			}
		}
		__syncthreads();      
	}
	if ((bx*BLOCK_SIZE_KRIGE1 + tx)<nx*ny*nz) prediction[bx*BLOCK_SIZE_KRIGE1 + tx] = sum + mean;	
}









/*******************************************************************************************
** UNCONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/

// global variables for unconditional simulation. These data are needed in the preprocessing as well as in generating realizations
struct uncond_state_3f {
	cufftComplex *d_cov; // d_cov is the result of the preprocessing ans is needed for each realozation
	int nx,ny,nz,n,m,o;
	float xmin,xmax,ymin,ymax,zmin,zmax,dx,dy,dz;
	int blockSize,numBlocks;
	dim3 blockSize2, numBlocks2;
	cufftHandle plan1;
	dim3 blockSize3d;
	dim3 blockCount3d;
	dim3 blockSize1d;
	dim3 blockCount1d;
} uncond_global_3f;


#ifdef __cplusplus
extern "C" {
#endif


void EXPORT unconditionalSimInit_3f(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, int *p_ny, float *p_zmin, float *p_zmax, int *p_nz, float *p_sill, float *p_range, float *p_nugget, int *p_covmodel, float *p_anis, int *do_check, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	
	uncond_global_3f.nx= *p_nx; // Number of cols
	uncond_global_3f.ny= *p_ny; // Number of rows
	uncond_global_3f.nz= *p_nz; // Number of z dims
	uncond_global_3f.n= 2*uncond_global_3f.nx; // Number of cols
	uncond_global_3f.m= 2*uncond_global_3f.ny; // Number of rows
	uncond_global_3f.o= 2*uncond_global_3f.nz; // Number of zdims
	//uncond_global_3f.n = ceil2(2*uncond_global_3f.nx); /// 
	//uncond_global_3f.m = ceil2(2*uncond_global_3f.ny); /// 
	//uncond_global_3f.o = ceil2(2*uncond_global_3f.nz); /// 
	uncond_global_3f.dx = (*p_xmax - *p_xmin) / (uncond_global_3f.nx-1);
	uncond_global_3f.dy = (*p_ymax - *p_ymin) / (uncond_global_3f.ny-1);
	uncond_global_3f.dz = (*p_zmax - *p_zmin) / (uncond_global_3f.nz-1);

	// 1d cuda grid
	uncond_global_3f.blockSize1d = dim3(256);
	uncond_global_3f.blockCount1d = dim3(uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o / uncond_global_3f.blockSize1d.x);
	if (uncond_global_3f.n * uncond_global_3f.m * uncond_global_3f.o % uncond_global_3f.blockSize1d.x  != 0) ++uncond_global_3f.blockCount1d.x;
	
	// 3d cuda grid
	uncond_global_3f.blockSize3d = dim3(8,8,4);
	uncond_global_3f.blockCount3d = dim3(uncond_global_3f.n / uncond_global_3f.blockSize3d.x, uncond_global_3f.m / uncond_global_3f.blockSize3d.y, uncond_global_3f.o / uncond_global_3f.blockSize3d.z);
	if (uncond_global_3f.n % uncond_global_3f.blockSize3d.x != 0) ++uncond_global_3f.blockCount3d.x;
	if (uncond_global_3f.m % uncond_global_3f.blockSize3d.y != 0) ++uncond_global_3f.blockCount3d.y;
	if (uncond_global_3f.o % uncond_global_3f.blockSize3d.z != 0) ++uncond_global_3f.blockCount3d.z;
	
	cufftPlan3d(&uncond_global_3f.plan1, uncond_global_3f.m, uncond_global_3f.n, uncond_global_3f.o, CUFFT_C2C); 

	
	// build grid (ROW MAJOR)
	// 3d grid:
	// z in [0,o-1] y in [0,n-1], x in [0,m-1]
	// --> f(x,y,z) = f[z*(n*m) + y*  ???????????????????ß


	/*cufftComplex *h_grid_c = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.m*uncond_global_3f.n*uncond_global_3f.o);
	for (int i=0; i<uncond_global_3f.n; ++i) { // i =  col index
		for (int j=0; j<uncond_global_3f.m; ++j) { // j = row index 
			h_grid_c[j*uncond_global_3f.n+i].x = *p_xmin + (i+1) * uncond_global_3f.dx; 
			h_grid_c[j*uncond_global_3f.n+i].y = *p_ymin + (j+1) * uncond_global_3f.dy;  
		}
	}*/
	
	float3 *h_grid_c = (float3 *)malloc(sizeof(float3)*uncond_global_3f.m*uncond_global_3f.n*uncond_global_3f.o);
	for (int k=0; k<uncond_global_3f.o; ++k) {
		for (int i=0; i<uncond_global_3f.n; ++i) { // i =  col index
			for (int j=0; j<uncond_global_3f.m; ++j) { // j = row index 
				h_grid_c[k*uncond_global_3f.n*uncond_global_3f.m + j*uncond_global_3f.n + i].x =  *p_xmin + (i+1) * uncond_global_3f.dx; 
				//h_grid_c[k*uncond_global_3f.n*uncond_global_3f.m + j*uncond_global_3f.n + i].y =  *p_ymin + (j+1) * uncond_global_3f.dy;
				h_grid_c[k*uncond_global_3f.n*uncond_global_3f.m + j*uncond_global_3f.n + i].y =  *p_ymin + (uncond_global_3f.m-1-j)* uncond_global_3f.dy;
				h_grid_c[k*uncond_global_3f.n*uncond_global_3f.m + j*uncond_global_3f.n + i].z =  *p_zmin + (k+1) * uncond_global_3f.dz; 
			}
		}
	}


	
	float xc = *p_xmin + (uncond_global_3f.dx*uncond_global_3f.n)/2;
	float yc = *p_ymin +(uncond_global_3f.dy*uncond_global_3f.m)/2;
	float zc = *p_ymin +(uncond_global_3f.dz*uncond_global_3f.o)/2;
	float sill = *p_sill;
	float range = *p_range;
	float nugget = *p_nugget;

	bool isotropic = (p_anis[3] == 1.0 && p_anis[4] == 1.0);
	float alpha = (90.0 - p_anis[0]) * (PI / 180.0);
	float beta = -1.0 * p_anis[1] * (PI / 180.0);
	float theta = p_anis[2] * (PI / 180.0);
	float afac1 = 1/p_anis[3];
	float afac2 = 1/p_anis[4];

	
	float3 *d_grid;
	
	// Array for grid
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(float3)*uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o);
	// Array for cov grid
	cudaStatus = cudaMalloc((void**)&uncond_global_3f.d_cov,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o);

	// Sample covariance and generate "trick" grid
	cufftComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o);
	
	// copy grid to GPU
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o*sizeof(cufftComplex),cudaMemcpyHostToDevice);

	if (isotropic) {
		sampleCovKernel_3f<<<uncond_global_3f.blockCount3d, uncond_global_3f.blockSize3d>>>(d_trick_grid_c, d_grid, uncond_global_3f.d_cov, xc, yc, zc,*p_covmodel, sill, range,nugget,uncond_global_3f.n,uncond_global_3f.m,uncond_global_3f.o);		
	}
	else {
		sampleCovAnisKernel_3f<<<uncond_global_3f.blockCount3d, uncond_global_3f.blockSize3d>>>(d_trick_grid_c, d_grid, uncond_global_3f.d_cov, xc, yc, zc,*p_covmodel, sill, range, nugget, alpha, beta, theta, afac1, afac2,uncond_global_3f.n,uncond_global_3f.m,uncond_global_3f.o);
	}
	free(h_grid_c);
	cudaFree(d_grid);
	 
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE COV MATRIX******* ///////
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_3f.d_cov,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\sampleCov.csv",h_cov,uncond_global_3f.m,uncond_global_3f.n);
//		free(h_cov);
//	}
//#endif

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE TRICK GRID ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\trickgrid.csv",h_cov,uncond_global_3f.m,uncond_global_3f.n);
//		free(h_cov);
//	}
//#endif
//


	// Execute 3d FFT of covariance grid in order to get the spectral representation 
	cufftExecC2C(uncond_global_3f.plan1, uncond_global_3f.d_cov, uncond_global_3f.d_cov, CUFFT_FORWARD); // in place fft forward


//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( COV GRID) ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_3f.d_cov,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftSampleCov.csv",h_cov,uncond_global_3f.m,uncond_global_3f.n);
//		free(h_cov);
//	}
//#endif
//
	cufftExecC2C(uncond_global_3f.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forward

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( TRICK GRID) ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftTrickGrid.csv",h_cov,uncond_global_3f.m,uncond_global_3f.n);
//		free(h_cov);
//	}
//#endif
	
	// Multiply fft of "trick" grid with n*m
	multKernel_3f<<<uncond_global_3f.blockCount1d, uncond_global_3f.blockSize1d>>>(d_trick_grid_c, uncond_global_3f.n, uncond_global_3f.m, uncond_global_3f.o);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel_3f!\n", cudaStatus);	

//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( TRICK GRID) ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m);
//		cudaStatus = cudaMemcpy(h_cov,d_trick_grid_c,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftTrickGridTimesNM.csv",h_cov,uncond_global_3f.m,uncond_global_3f.n);
//		free(h_cov);
//	}
//#endif


	// Devide spectral covariance grid by "trick" grid
	divideSpectrumKernel_3f<<<uncond_global_3f.blockCount1d, uncond_global_3f.blockSize1d>>>(uncond_global_3f.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel_f!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	
//#ifdef DEBUG 
//	{
//		/// ****** TEST AUSGABE FFT( COV GRID) / FFT(TRICKGRID)*N*M ******* /////// 
//		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m);
//		cudaStatus = cudaMemcpy(h_cov,uncond_global_3f.d_cov,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m,cudaMemcpyDeviceToHost);
//		writeCSVMatrix("C:\\fft\\fftSampleCovByTrickGridNM.csv",h_cov,uncond_global_3f.m,uncond_global_3f.n);
//		free(h_cov);
//	}
//#endif
//


	// Copy to host and check for negative real parts
	if (*do_check) {
		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o);
		cudaStatus = cudaMemcpy(h_cov,uncond_global_3f.d_cov,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o,cudaMemcpyDeviceToHost);
		for (int i=0; i<uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o; ++i) {
			if (h_cov[i].x < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				free(h_cov);
				cudaFree(uncond_global_3f.d_cov);
				cufftDestroy(uncond_global_3f.plan1);
				return;
			}	
		}
		free(h_cov);
	}

	// Compute sqrt of cov grid
	sqrtKernel_3f<<<uncond_global_3f.blockCount1d, uncond_global_3f.blockSize1d>>>(uncond_global_3f.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel_f\n", cudaStatus);

}

// Generates unconditional realizations
// p_out = output array of size nx*ny*k * sizeof(float)
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT unconditionalSimRealizations_3f(float *p_out,  int *p_k, int *ret_code)
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

	cudaStatus = cudaMalloc((void**)&d_rand,sizeof(float)*uncond_global_3f.m*uncond_global_3f.n*uncond_global_3f.o); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);

	cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftComplex) * uncond_global_3f.n * uncond_global_3f.m * uncond_global_3f.o); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);

	dim3 blockSize3dhalf  = dim3(8,8,4);
	dim3 blockCount3dhalf = dim3(uncond_global_3f.nx/blockSize3dhalf.x,uncond_global_3f.ny/blockSize3dhalf.y,uncond_global_3f.nz/blockSize3dhalf.z);
	if (uncond_global_3f.nx % blockSize3dhalf.x != 0) ++blockCount3dhalf.x;
	if (uncond_global_3f.ny % blockSize3dhalf.y != 0) ++blockCount3dhalf.y;
	if (uncond_global_3f.nz % blockSize3dhalf.z != 0) ++blockCount3dhalf.z;

	for(int l = 0; l<k; ++l) {
				
		// Generate Random Numbers
		curandGenerateNormal(curandGen,d_rand,uncond_global_3f.m*uncond_global_3f.n*uncond_global_3f.o,0.0,1.0);
		
		// Convert real random numbers to complex numbers
		realToComplexKernel_3f<<< uncond_global_3f.blockCount1d, uncond_global_3f.blockSize1d>>>(d_fftrand, d_rand, uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel_f!\n", cudaStatus);	

		// Compute 2D FFT of random numbers
		cufftExecC2C(uncond_global_3f.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward

		if(l==0) cudaMalloc((void**)&d_amp,sizeof(cufftComplex)*uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o);
		elementProduct_3f<<<uncond_global_3f.blockCount1d, uncond_global_3f.blockSize1d>>>(d_amp, uncond_global_3f.d_cov, d_fftrand, uncond_global_3f.m*uncond_global_3f.n*uncond_global_3f.o);  
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct_f!\n", cudaStatus);

		cufftExecC2C(uncond_global_3f.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation		
		if(l==0) cudaMalloc((void**)&d_out,sizeof(float)*uncond_global_3f.nx*uncond_global_3f.ny*uncond_global_3f.nz);
		
		
		ReDiv_3f<<<blockCount3dhalf, blockSize3dhalf>>>(d_out, d_amp, std::sqrt((float)(uncond_global_3f.n*uncond_global_3f.m*uncond_global_3f.o)), uncond_global_3f.nx, uncond_global_3f.ny, uncond_global_3f.nz);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess) {
			printf("cudaThreadSynchronize returned error code %d after launching ReDiv_3f!\n", cudaStatus);
		}
		cudaMemcpy((p_out + l*(uncond_global_3f.nx*uncond_global_3f.ny*uncond_global_3f.nz)),d_out,sizeof(float)*uncond_global_3f.nx*uncond_global_3f.ny*uncond_global_3f.nz,cudaMemcpyDeviceToHost);
	}

	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_out);
	curandDestroyGenerator(curandGen);
}


void EXPORT unconditionalSimRelease_3f(int *ret_code) {
	*ret_code = OK;
	cudaFree(uncond_global_3f.d_cov);
	cufftDestroy(uncond_global_3f.plan1);
}


#ifdef __cplusplus
}
#endif
















/*******************************************************************************************
** CONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/


// global variables for conditional simulation that are needed both, for initialization as well as for generating realizations
struct cond_state_3f {
	cufftComplex *d_cov; 
	int nx,ny,nz,n,m,o;
	float xmin,xmax,zmin,zmax,ymin,ymax,dx,dy,dz;
	float range, sill, nugget;
	float alpha,beta,theta,afac1,afac2;
	bool isotropic;
	int blockSize,numBlocks;
	dim3 blockSize2, numBlocks2;
	cufftHandle plan1;
	dim3 blockSize3d;
	dim3 blockCount3d;
	dim3 blockSize1d;
	dim3 blockCount1d;
	dim3 blockSizeSamples;
	dim3 blockCountSamples;
	dim3 blockSizeSamplesPlus1;
	dim3 blockCountSamplesPlus1;
	// Variables for conditioning
	int numSrc; // Number of sample observation
	float3 *d_samplexy; // coordinates of samples
	float3 *d_sampleindices; // Corresponding grid indices in subpixel accuracy
	float *d_sampledata; // data values of samples
	//float *d_covinv; // inverse covariance matrix of samples
	float *d_uncond;
	int covmodel;
	int k;
	float mu; // known mean for simple kriging
	int krige_method;
} cond_global_3f;









#ifdef __cplusplus
extern "C" {
#endif





void EXPORT conditionalSimInit_3f(float *p_xmin, float *p_xmax, int *p_nx, float *p_ymin, float *p_ymax, 
								  int *p_ny, float *p_zmin, float *p_zmax, int *p_nz, float *p_sill, float *p_range, 
								  float *p_nugget, float *p_srcXY,  float *p_srcData, int *p_numSrc, int *p_covmodel, 
								  float *p_anis, int *do_check, int *krige_method, float *mu, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	cublasInit();

	cond_global_3f.nx= *p_nx; // Number of cols
	cond_global_3f.ny= *p_ny; // Number of rows
	cond_global_3f.nz= *p_nz; // Number of rows
	cond_global_3f.n= 2*cond_global_3f.nx; // Number of cols
	cond_global_3f.m= 2*cond_global_3f.ny; // Number of rows
	cond_global_3f.o= 2*cond_global_3f.ny; // Number of rows
	cond_global_3f.dx = (*p_xmax - *p_xmin) / (cond_global_3f.nx - 1);
	cond_global_3f.dy = (*p_ymax - *p_ymin) / (cond_global_3f.ny - 1);
	cond_global_3f.dz = (*p_zmax - *p_zmin) / (cond_global_3f.nz - 1);
	cond_global_3f.numSrc = *p_numSrc;
	cond_global_3f.xmin = *p_xmin;
	cond_global_3f.xmax = *p_xmax;
	cond_global_3f.ymin = *p_ymin;
	cond_global_3f.ymax = *p_ymax;
	cond_global_3f.zmin = *p_zmin;
	cond_global_3f.zmax = *p_zmax;
	cond_global_3f.range = *p_range;
	cond_global_3f.sill = *p_sill;
	cond_global_3f.nugget = *p_nugget;
	cond_global_3f.covmodel = *p_covmodel;
	cond_global_3f.krige_method = *krige_method;
	if (cond_global_3f.krige_method == SIMPLE)
		cond_global_3f.mu = *mu;
	else cond_global_3f.mu = 0;

	cond_global_3f.isotropic = (p_anis[3] == 1.0 && p_anis[4] == 1.0);
	cond_global_3f.alpha = (90.0 - p_anis[0]) * (PI / 180.0);
	cond_global_3f.beta = -1.0 * p_anis[1] * (PI / 180.0);
	cond_global_3f.theta = p_anis[2] * (PI / 180.0);
	cond_global_3f.afac1 = 1/p_anis[3];
	cond_global_3f.afac2 = 1/p_anis[4];



	// 1d cuda grid
	cond_global_3f.blockSize1d = dim3(256);
	cond_global_3f.blockCount1d = dim3(cond_global_3f.n*cond_global_3f.m*cond_global_3f.o / cond_global_3f.blockSize1d.x);
	if (cond_global_3f.n * cond_global_3f.m * cond_global_3f.o % cond_global_3f.blockSize1d.x  != 0) ++cond_global_3f.blockCount1d.x;
	
	// 3d cuda grid
	cond_global_3f.blockSize3d = dim3(8,8,4);
	cond_global_3f.blockCount3d = dim3(cond_global_3f.n / cond_global_3f.blockSize3d.x, cond_global_3f.m / cond_global_3f.blockSize3d.y, cond_global_3f.o / cond_global_3f.blockSize3d.z);
	if (cond_global_3f.n % cond_global_3f.blockSize3d.x != 0) ++cond_global_3f.blockCount3d.x;
	if (cond_global_3f.m % cond_global_3f.blockSize3d.y != 0) ++cond_global_3f.blockCount3d.y;
	if (cond_global_3f.o % cond_global_3f.blockSize3d.z != 0) ++cond_global_3f.blockCount3d.z;
	
	cufftPlan3d(&cond_global_3f.plan1, cond_global_3f.m, cond_global_3f.n, cond_global_3f.o, CUFFT_C2C); 

	
	// build grid (ROW MAJOR)

	// 1d cuda grid for samples
	cond_global_3f.blockSizeSamples = dim3(256);
	cond_global_3f.blockCountSamples = dim3(cond_global_3f.numSrc / cond_global_3f.blockSizeSamples.x);
	if (cond_global_3f.numSrc % cond_global_3f.blockSizeSamples.x !=0) ++cond_global_3f.blockCountSamples.x;

	
	

	float3 *h_grid_c = (float3 *)malloc(sizeof(float3)*cond_global_3f.m*cond_global_3f.n*cond_global_3f.o);
	for (int k=0; k<cond_global_3f.o; ++k) {
		for (int i=0; i<cond_global_3f.n; ++i) { // i =  col index
			for (int j=0; j<cond_global_3f.m; ++j) { // j = row index 
				h_grid_c[k*cond_global_3f.n*cond_global_3f.m + j*cond_global_3f.n + i].x =  *p_xmin + (i+1) * cond_global_3f.dx; 
				//h_grid_c[k*cond_global_3f.n*cond_global_3f.m + j*cond_global_3f.n + i].y =  *p_ymin + (j+1) * cond_global_3f.dy;
				h_grid_c[k*cond_global_3f.n*cond_global_3f.m + j*cond_global_3f.n + i].y =  *p_ymin + (cond_global_3f.m-1-j)* cond_global_3f.dy;
				h_grid_c[k*cond_global_3f.n*cond_global_3f.m + j*cond_global_3f.n + i].z =  *p_zmin + (k+1) * cond_global_3f.dz; 
			}
		}
	}

	float xc = *p_xmin + (cond_global_3f.dx*cond_global_3f.n)/2;
	float yc = *p_ymin +(cond_global_3f.dy*cond_global_3f.m)/2;
	float zc = *p_ymin +(cond_global_3f.dz*cond_global_3f.o)/2;
	
	float3 *d_grid;
	
	// Allocate grid and cov arrays on GPU
	cudaStatus = cudaMalloc((void**)&d_grid,sizeof(float3)*cond_global_3f.n*cond_global_3f.m*cond_global_3f.o);
	cudaStatus = cudaMalloc((void**)&cond_global_3f.d_cov,sizeof(cufftComplex)*cond_global_3f.n*cond_global_3f.m*cond_global_3f.o);

	// Sample covariance and generate "trick" grid
	cufftComplex *d_trick_grid_c;
	cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(cufftComplex)*cond_global_3f.n*cond_global_3f.m*cond_global_3f.o);
	
	// copy grid to GPU
	cudaStatus = cudaMemcpy(d_grid,h_grid_c, cond_global_3f.n*cond_global_3f.m*cond_global_3f.o*sizeof(cufftComplex),cudaMemcpyHostToDevice);

	if (cond_global_3f.isotropic) {
		sampleCovKernel_3f<<<cond_global_3f.blockCount3d, cond_global_3f.blockSize3d>>>(d_trick_grid_c, d_grid, cond_global_3f.d_cov, xc, yc, zc,*p_covmodel, cond_global_3f.sill, cond_global_3f.range,cond_global_3f.nugget,cond_global_3f.n,cond_global_3f.m,cond_global_3f.o);		
	}
	else {
		sampleCovAnisKernel_3f<<<cond_global_3f.blockCount3d, cond_global_3f.blockSize3d>>>(d_trick_grid_c, d_grid, cond_global_3f.d_cov, xc, yc, zc,*p_covmodel, cond_global_3f.sill, cond_global_3f.range, cond_global_3f.nugget, cond_global_3f.alpha, cond_global_3f.beta, cond_global_3f.theta, cond_global_3f.afac1, cond_global_3f.afac2,cond_global_3f.n,cond_global_3f.m,cond_global_3f.o);
	}
	free(h_grid_c);
	cudaFree(d_grid);



	// Compute spectral representation of cov and "trick" grid
	cufftExecC2C(cond_global_3f.plan1, cond_global_3f.d_cov, cond_global_3f.d_cov, CUFFT_FORWARD); // in place fft forward
	cufftExecC2C(cond_global_3f.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forwar


	// Multiplication of fft(trick_grid) with n*m	
	multKernel_3f<<<cond_global_3f.blockCount1d, cond_global_3f.blockSize1d>>>(d_trick_grid_c, cond_global_3f.n, cond_global_3f.m, cond_global_3f.o);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching multKernel_3f!\n", cudaStatus);	

	// Devide spectral cov grid by fft of "trick" grid
	divideSpectrumKernel_3f<<<cond_global_3f.blockCount1d, cond_global_3f.blockSize1d>>>(cond_global_3f.d_cov, d_trick_grid_c);	
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching divideSpectrumKernel_f!\n", cudaStatus);	
	cudaFree(d_trick_grid_c);

	// Copy to host and check for negative real parts
	if (*do_check) {
		cufftComplex *h_cov = (cufftComplex*)malloc(sizeof(cufftComplex)*cond_global_3f.n*cond_global_3f.m*cond_global_3f.o);
		cudaStatus = cudaMemcpy(h_cov,cond_global_3f.d_cov,sizeof(cufftComplex)*cond_global_3f.n*cond_global_3f.m*cond_global_3f.o,cudaMemcpyDeviceToHost);
		for (int i=0; i<cond_global_3f.n*cond_global_3f.m*cond_global_3f.o; ++i) {
			if (h_cov[i].x < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				free(h_cov);
				cudaFree(cond_global_3f.d_cov);
				cufftDestroy(cond_global_3f.plan1);
				return;
			}	
		}
		free(h_cov);
	}



	// Compute sqrt of spectral cov grid
	sqrtKernel_3f<<<cond_global_3f.blockCount1d, cond_global_3f.blockSize1d>>>(cond_global_3f.d_cov);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sqrtKernel_f\n", cudaStatus);

	// Copy samples to gpu
	cudaStatus = cudaMalloc((void**)&cond_global_3f.d_samplexy,sizeof(float3)* cond_global_3f.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global_3f.d_sampleindices,sizeof(float3)*cond_global_3f.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&cond_global_3f.d_sampledata,sizeof(float)*cond_global_3f.numSrc); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMemcpy(cond_global_3f.d_samplexy,p_srcXY,sizeof(float3)* cond_global_3f.numSrc,cudaMemcpyHostToDevice);
	cudaMemcpy(cond_global_3f.d_sampledata,p_srcData,sizeof(float)*cond_global_3f.numSrc,cudaMemcpyHostToDevice);
		

	// Overlay samples to grid and save resulting subpixel grind indices
	overlay_3f<<<cond_global_3f.blockCountSamples,cond_global_3f.blockSizeSamples>>>(cond_global_3f.d_sampleindices,cond_global_3f.d_samplexy,*p_xmin,cond_global_3f.dx,*p_ymax,cond_global_3f.dy,cond_global_3f.zmin, cond_global_3f.dz, cond_global_3f.numSrc);
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching overlay_3f!\n", cudaStatus);	
	// Warning: It is not checked, whether sample points truly lie inside the grid's boundaries. This may lead to illegal memory access			

	/* TEST OUTPUT ON HOST */
	/*float2 *h_indices = (float2*)malloc(sizeof(float2)*cond_global_3f.numSrc);
	cudaMemcpy(h_indices,cond_global_3f.d_sampleindices,sizeof(float2)*cond_global_3f.numSrc,cudaMemcpyDeviceToHost);
	for (int i=0;i<cond_global_3f.numSrc;++i) {
		printf("(%.2f,%.2f) -> (%.2f,%.2f)\n",p_srcXY[2*i],p_srcXY[2*i+1],h_indices[i].x, h_indices[i].y);
	}
	free(h_indices);*/
}




// Generates Unconditional Realizations and the residuals of all samples to all realizations 
// p_out = output matrix of residuals, col means number of realization, row represents a sample point
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT conditionalSimUncondResiduals_3f(float *p_out, int *p_k, int *ret_code) {
	*ret_code = OK;
	cudaError_t cudaStatus;
	cond_global_3f.k = *p_k;
	
	float *d_rand; // Device Random Numbers
	curandGenerator_t curandGen;
	cufftComplex *d_fftrand;
	cufftComplex* d_amp;	
	float *d_residuals; // residuals of samples and underlying unconditional realization
	
	curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGen,(unsigned long long)(time(NULL)));	
	
	cudaStatus = cudaMalloc((void**)&d_rand,sizeof(float)*cond_global_3f.m*cond_global_3f.n*cond_global_3f.o); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaStatus = cudaMalloc((void**)&d_fftrand,sizeof(cufftComplex) * cond_global_3f.n * cond_global_3f.m * cond_global_3f.o); 
	if (cudaStatus != cudaSuccess)  printf("cudaMalloc returned error code %d\n", cudaStatus);
	cudaMalloc((void**)&d_amp,sizeof(cufftComplex)*cond_global_3f.n*cond_global_3f.m*cond_global_3f.o);
	cudaMalloc((void**)&cond_global_3f.d_uncond,sizeof(float)*cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz * cond_global_3f.k);
	
	
	if (cond_global_3f.krige_method == ORDINARY) {
		cudaMalloc((void**)&d_residuals,sizeof(float)* (cond_global_3f.numSrc + 1));
	}
	else if (cond_global_3f.krige_method == SIMPLE) {
		cudaMalloc((void**)&d_residuals,sizeof(float)* cond_global_3f.numSrc);
	}
		
	for(int l=0; l<cond_global_3f.k; ++l) {
			
		
		curandGenerateNormal(curandGen,d_rand,cond_global_3f.m*cond_global_3f.n*cond_global_3f.o,0.0,1.0);	
		realToComplexKernel_3f<<< cond_global_3f.blockCount1d, cond_global_3f.blockSize1d>>>(d_fftrand, d_rand, cond_global_3f.n*cond_global_3f.m*cond_global_3f.o);
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching realToComplexKernel_f!\n", cudaStatus);	
		cufftExecC2C(cond_global_3f.plan1, d_fftrand, d_fftrand, CUFFT_FORWARD); // in place fft forward
		cudaStatus = cudaThreadSynchronize();
		
		elementProduct_3f<<<cond_global_3f.blockCount1d, cond_global_3f.blockSize1d>>>(d_amp, cond_global_3f.d_cov, d_fftrand, cond_global_3f.m*cond_global_3f.n*cond_global_3f.o);
    
		cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching elementProduct_f!\n", cudaStatus);

		cufftExecC2C(cond_global_3f.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation
	  
		dim3 blockSize3dhalf  = dim3(8,8,4);
		dim3 blockCount3dhalf = dim3(cond_global_3f.nx/blockSize3dhalf.x,cond_global_3f.ny/blockSize3dhalf.y,cond_global_3f.nz/blockSize3dhalf.z);
		if (cond_global_3f.nx % blockSize3dhalf.x != 0) ++blockCount3dhalf.x;
		if (cond_global_3f.ny % blockSize3dhalf.y != 0) ++blockCount3dhalf.y;
		if (cond_global_3f.nz % blockSize3dhalf.z != 0) ++blockCount3dhalf.z;
		ReDiv_3f<<<blockCount3dhalf, blockSize3dhalf>>>(cond_global_3f.d_uncond + l*cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz, d_amp, std::sqrt((float)(cond_global_3f.n*cond_global_3f.m*cond_global_3f.o)), cond_global_3f.nx, cond_global_3f.ny, cond_global_3f.nz);
		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess) printf("cudaThreadSynchronize returned error code %d after launching ReDiv_3f!\n", cudaStatus);
		
		// d_uncond is now a unconditional realization 
		// Compute residuals between samples and d_uncond
		if (cond_global_3f.krige_method == ORDINARY) {
			residualsOrdinary_3f<<<cond_global_3f.blockCountSamples,cond_global_3f.blockSizeSamples>>>(d_residuals,cond_global_3f.d_sampledata,cond_global_3f.d_uncond+l*(cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz),cond_global_3f.d_sampleindices,cond_global_3f.nx,cond_global_3f.ny,cond_global_3f.nz,cond_global_3f.numSrc);
		}
		else if (cond_global_3f.krige_method == SIMPLE) {
			residualsSimple_3f<<<cond_global_3f.blockCountSamples,cond_global_3f.blockSizeSamples>>>(d_residuals,cond_global_3f.d_sampledata,cond_global_3f.d_uncond+l*(cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz),cond_global_3f.d_sampleindices,cond_global_3f.nx,cond_global_3f.ny,cond_global_3f.nz,cond_global_3f.numSrc, cond_global_3f.mu);
		}


		cudaStatus = cudaThreadSynchronize();	
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching residuals!\n", cudaStatus);
	

		// Copy residuals to R, col major...
		if (cond_global_3f.krige_method == ORDINARY) {
			cudaMemcpy((p_out + l*(cond_global_3f.numSrc + 1)),d_residuals,sizeof(float)* (cond_global_3f.numSrc + 1),cudaMemcpyDeviceToHost);	
		}
		else if (cond_global_3f.krige_method == SIMPLE) {
			cudaMemcpy(p_out + l*cond_global_3f.numSrc,d_residuals,sizeof(float) * cond_global_3f.numSrc,cudaMemcpyDeviceToHost);	
		}
	}
	curandDestroyGenerator(curandGen);
	
	cudaFree(d_rand);
	cudaFree(d_fftrand);
	cudaFree(d_amp);
	cudaFree(d_residuals);
}


void EXPORT conditionalSimKrigeResiduals_3f(float *p_out, float *p_y, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus = cudaSuccess;
	
	float *d_y; // result vector from solving the kriging equation system
	float *d_respred; // interpolated residuals
	cudaMalloc((void**)&d_y, sizeof(float) * (cond_global_3f.numSrc + 1));
	cudaMalloc((void**)&d_respred, sizeof(float) * cond_global_3f.nx * cond_global_3f.ny  * cond_global_3f.nz);
	
	dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
	dim3 blockCntKrige = dim3((cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz) / blockSizeKrige.x);
	if ((cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz) % blockSizeKrige.x != 0) ++blockCntKrige.x;
	
	dim3 blockSizeCond = dim3(256);
	dim3 blockCntCond = dim3(cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz/ blockSizeCond.x);
	if (cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz % blockSizeCond.x != 0) ++blockSizeCond.x;

	for(int l = 0; l<cond_global_3f.k; ++l) {
						
		
		cudaMemcpy(d_y, p_y + l*(cond_global_3f.numSrc + 1), sizeof(float) * (cond_global_3f.numSrc + 1),cudaMemcpyHostToDevice);		
		
		// Kriging prediction
		if (cond_global_3f.isotropic)
			krigingKernel_3f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_3f.d_samplexy,cond_global_3f.xmin,cond_global_3f.dx,cond_global_3f.ymin,cond_global_3f.dy,cond_global_3f.zmin,cond_global_3f.dz,d_y,cond_global_3f.covmodel,cond_global_3f.range,cond_global_3f.sill,cond_global_3f.nugget,cond_global_3f.numSrc,cond_global_3f.nx,cond_global_3f.ny,cond_global_3f.nz);
		else 	
			krigingAnisKernel_3f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_3f.d_samplexy,cond_global_3f.xmin,cond_global_3f.dx,cond_global_3f.ymin,cond_global_3f.dy,cond_global_3f.zmin,cond_global_3f.dz,d_y,cond_global_3f.covmodel,cond_global_3f.range,cond_global_3f.sill,cond_global_3f.nugget, cond_global_3f.alpha, cond_global_3f.beta, cond_global_3f.theta, cond_global_3f.afac1, cond_global_3f.afac2, cond_global_3f.numSrc,cond_global_3f.nx,cond_global_3f.ny,cond_global_3f.nz);
		

		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_3f!\n", cudaStatus);
		
		// Add result to unconditional realization
		addResSim_3f<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_3f.d_uncond + l*cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz, cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz);
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_f!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz)),d_respred,sizeof(float)*cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz,cudaMemcpyDeviceToHost);		
		
	}
	cudaFree(d_y);
	cudaFree(d_respred);
}




void EXPORT conditionalSimSimpleKrigeResiduals_3f(float *p_out, float *p_y, int *ret_code)
{
	*ret_code = OK;
	cudaError_t cudaStatus = cudaSuccess;
	
	float *d_y; // result vector from solving the kriging equation system
	float *d_respred; // interpolated residuals
	cudaMalloc((void**)&d_y, sizeof(float) * cond_global_3f.numSrc); // not + 1, no lagrange multiplicator in simple kriging
	cudaMalloc((void**)&d_respred, sizeof(float) * cond_global_3f.nx * cond_global_3f.ny *cond_global_3f.nz);
	
	dim3 blockSizeKrige = dim3(BLOCK_SIZE_KRIGE1);
	dim3 blockCntKrige = dim3((cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz) / blockSizeKrige.x);
	if ((cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz) % blockSizeKrige.x != 0) ++blockCntKrige.x;
	
	dim3 blockSizeCond = dim3(256);
	dim3 blockCntCond = dim3(cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz/ blockSizeCond.x);
	if (cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz % blockSizeCond.x != 0) ++blockSizeCond.x;

	for(int l = 0; l<cond_global_3f.k; ++l) {
						
		cudaMemcpy(d_y, p_y + l*cond_global_3f.numSrc, sizeof(float) * cond_global_3f.numSrc,cudaMemcpyHostToDevice);	// not + 1, no lagrange multiplicator in simple kriging		
		// Kriging prediction
		if (cond_global_3f.isotropic)
			krigingSimpleKernel_3f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_3f.d_samplexy,cond_global_3f.xmin,cond_global_3f.dx,cond_global_3f.ymin,cond_global_3f.dy,cond_global_3f.zmin,cond_global_3f.dz,d_y,cond_global_3f.covmodel,cond_global_3f.range,cond_global_3f.sill,cond_global_3f.nugget,cond_global_3f.numSrc,cond_global_3f.nx,cond_global_3f.ny,cond_global_3f.nz,cond_global_3f.mu);
		else 	
			krigingSimpleAnisKernel_3f<<<blockCntKrige, blockSizeKrige>>>(d_respred,cond_global_3f.d_samplexy,cond_global_3f.xmin,cond_global_3f.dx,cond_global_3f.ymin,cond_global_3f.dy,cond_global_3f.zmin,cond_global_3f.dz,d_y,cond_global_3f.covmodel,cond_global_3f.range,cond_global_3f.sill,cond_global_3f.nugget, cond_global_3f.alpha, cond_global_3f.beta, cond_global_3f.theta, cond_global_3f.afac1, cond_global_3f.afac2, cond_global_3f.numSrc,cond_global_3f.nx,cond_global_3f.ny,cond_global_3f.nz,cond_global_3f.mu);
		

		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching krigingExpKernel_3f!\n", cudaStatus);
		
		// Add result to unconditional realization
		addResSim_3f<<<blockCntCond,blockSizeCond>>>(d_respred, cond_global_3f.d_uncond + l*cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz, cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz);
		if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching addResSim_f!\n", cudaStatus);
		
		// Write Result to R
		cudaMemcpy((p_out + l*(cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz)),d_respred,sizeof(float)*cond_global_3f.nx*cond_global_3f.ny*cond_global_3f.nz,cudaMemcpyDeviceToHost);		
		
	}
	cudaFree(d_y);
	cudaFree(d_respred);
}





void EXPORT conditionalSimRelease_3f(int *ret_code) {
	*ret_code = OK;
	cufftDestroy(cond_global_3f.plan1);
	cudaFree(cond_global_3f.d_samplexy);
	cudaFree(cond_global_3f.d_sampledata);
	cudaFree(cond_global_3f.d_sampleindices);
	cudaFree(cond_global_3f.d_cov);
	cudaFree(cond_global_3f.d_uncond);
}




#ifdef __cplusplus
}
#endif










