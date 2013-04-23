


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



//#define DEBUG
/*******************************************************************************************
** GPU KERNELS *****************************************************************************
********************************************************************************************/

inline double covExpKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0)? (nugget + sill) : (sill*exp(-dist/range)));
}



inline double covExpAnisKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	return ((dist == 0.0)? (nugget + sill) : (sill*exp(-dist/range)));
}




inline double covGauKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist2 = (ax-bx)*(ax-bx)+(ay-by)*(ay-by);
	return ((dist2 == 0.0)? (nugget + sill) : (sill*exp(-dist2/(range*range))));
}



inline double covGauAnisKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	//dist = sqrtf(dist);

	return ((dist == 0.0)? (nugget + sill) : (sill*exp(-dist/(range*range))));
}


inline double covSphKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	if (dist == 0.0) 
		return(nugget + sill);	
	else if(dist <= range) 
		return sill * (1.0 - (((3.0*dist) / (2.0*range)) - ((dist * dist * dist) / (2.0 * range * range * range)) ));	
	return 0.0; // WARNING,  sample cov matrix may be not regular for wenn point pairs with distance > range
}



inline double covSphAnisKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
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



inline double covMat3Kernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*exp(-SQRT3*dist/range)));
}


inline double covMat3AnisKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;

	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	return ((dist == 0.0)? (nugget + sill) : (sill*(1+SQRT3*dist/range)*exp(-SQRT3*dist/range)));
}


inline double covMat5Kernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget) {
	double dist = sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by));
	return ((dist == 0.0)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * exp(-SQRT5*dist/range)));
}


inline double covMat5AnisKernel_cpu_2d(double ax, double ay, double bx, double by, double sill, double range, double nugget, double alpha, double afac1) {
	double dist = 0.0;
	double temp = 0.0;
	double dx = ax-bx;
	double dy = ay-by;
	
	temp = dx * cos(alpha) + dy * sin(alpha);
	dist += temp * temp;
	temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
	dist += temp * temp;
	dist = sqrt(dist);

	return ((dist == 0.0)? (nugget + sill) : (sill * (1 + SQRT5*dist/range + 5*dist*dist/3*range*range) * exp(-SQRT5*dist/range)));
}




// Converts real double array into fftw_complex array
void realToComplexKernel_cpu_2d(fftw_complex *c, double* r, int n) {
	
	for (int i=0; i<n; ++i)
	{
		c[i][0] = r[i];
		c[i][1] = 0.0;
	}
}

void ReDiv_cpu_2d(double *out, fftw_complex *c, double div,int nx, int ny, int M) {
	for (int col=0; col<nx; ++col) {
		for (int row=0; row<ny; ++row) {
			out[row*nx+col] = c[row*M+col][0] / div; 
		}
	}
}





// Covariance sampling of a regular grid
void sampleCovKernel_cpu_2d(fftw_complex *trickgrid, fftw_complex *grid, fftw_complex* cov, double xc, double yc, int model, double sill, double range, double nugget, int n, int m) {

	for (int col=0; col<n; ++col) {
		for (int row=0; row<m; ++row) {
			switch (model) {
			case EXP:
				cov[row*n+col][0] = covExpKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget);
				break;
			case GAU:
				cov[row*n+col][0] = covGauKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget);
				break;
			case SPH:
				cov[row*n+col][0] = covSphKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget);
				break;
			case MAT3:
				cov[row*n+col][0] = covMat3Kernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget);
				break;
			case MAT5:
				cov[row*n+col][0] = covMat5Kernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget);
				break;
			}	
			cov[row*n+col][1] = 0.0;
			trickgrid[row*n+col][0] = 0.0;
			trickgrid[row*n+col][1] = 0.0;	
		}
	}
	trickgrid[(m/2-1)*n+(n/2-1)][0] = 1.0;
}




void sampleCovAnisKernel_cpu_2d(fftw_complex *trickgrid, fftw_complex *grid, fftw_complex* cov, double xc, double yc, int model, double sill, double range, double nugget, double alpha, double afac1, int n, int m) {
	for (int col=0; col<n; ++col) {
		for (int row=0; row<m; ++row) {
			switch (model) {
			case EXP:
				cov[row*n+col][0] = covExpAnisKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget,alpha,afac1);
				break;
			case GAU:
				cov[row*n+col][0] = covGauAnisKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget,alpha,afac1);
				break;
			case SPH:
				cov[row*n+col][0] = covSphAnisKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget,alpha,afac1);
				break;
			case MAT3:
				cov[row*n+col][0] = covMat3AnisKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget,alpha,afac1);
				break;
			case MAT5:
				cov[row*n+col][0] = covMat5AnisKernel_cpu_2d(grid[row*n+col][0],grid[row*n+col][1],xc,yc,sill,range,nugget,alpha,afac1);
				break;
			}	
			cov[row*n+col][1] = 0.0;
			trickgrid[row*n+col][0] = 0.0;
			trickgrid[row*n+col][1] = 0.0;	
		}
	}
	trickgrid[(m/2-1)*n+(n/2-1)][0] = 1.0;
}
	




// TODO: GRID DIMENSIONS???
void multKernel_cpu_2d(fftw_complex *fftgrid, int n, int m) {
	for (int i=0; i<n*m; ++i) {
		fftgrid[i][0] = fftgrid[i][0]*n*m;
		fftgrid[i][1] = fftgrid[i][1]*n*m;
	}
}


void divideSpectrumKernel_cpu_2d(fftw_complex *spectrum, fftw_complex *fftgrid, int n, int m, double eigenvals_tol) {
	for (int i=0; i<n*m; ++i) {
		double a = spectrum[i][0];
		double b = spectrum[i][1];
		double c = fftgrid[i][0];
		double d = fftgrid[i][1];
		spectrum[i][0] = (a*c+b*d)/(c*c+d*d);
		if (spectrum[i][0] < 0 && spectrum[i][0] > eigenvals_tol) spectrum[i][0] = 0.0;
		spectrum[i][1] = (b*c-a*d)/(c*c+d*d);
	}
}

// Sets negative real parts of cov grid to 0
 void setCov0Kernel_cpu_2d(fftw_complex *cov, int nm)
{
	for (int i = 0; i < nm; ++i) 
	{
		if (i < nm) {
			if (cov[i][0] < 0.0) cov[i][0] = 0.0;
		}
	}
}





void sqrtKernel_cpu_2d(fftw_complex *spectrum, int n, int m) {
	for (int i=0; i<n*m; ++i) {
		double re = spectrum[i][0];
		double im = spectrum[i][1];
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
		spectrum[i][0] = dsqrt*cos(sill/2);
		spectrum[i][1] = dsqrt*sin(sill/2);
	}
}




 void elementProduct_cpu_2d(fftw_complex *c, fftw_complex *a, fftw_complex *b, int n) {
	for (int i=0; i<n; ++i) {
		c[i][0] = a[i][0] * b[i][0] - a[i][1] * b[i][1];
		c[i][1] = a[i][0] * b[i][1] + a[i][1] * b[i][0];
	}
}



void krigingKernel_cpu_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, int numSrc, int nx, int ny)
{	
	for (int col=0; col < nx; ++col) {
		for (int row=0; row < ny; ++row) {
			double sum=0.0;
			double yr_x = col * dx + xmin;
			double yr_y = (ny - 1 - row) * dy + ymin;

			for (int i=0; i < numSrc; ++i) {
				switch (model) {
					case EXP:
						sum += covExpKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case GAU:
						sum += covGauKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case SPH:
						sum += covSphKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					}

			}
			prediction[row * nx + col] = sum + y[numSrc]; // TODO: TEST!
		}

	}
}



void krigingAnisKernel_cpu_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, double alpha, double afac1, int numSrc, int nx, int ny)
{	
	for (int col=0; col < nx; ++col) {
		for (int row=0; row < ny; ++row) {
			double sum=0.0;
			double yr_x = col * dx + xmin;
			double yr_y = (ny - 1 - row) * dy + ymin;

			for (int i=0; i < numSrc; ++i) {
				switch (model) {
					case EXP:
						sum += covExpAnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case GAU:
						sum += covGauAnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					}

			}
			prediction[row * nx + col] = sum + y[numSrc]; // TODO: TEST!
		}

	}
}




void krigingSimpleKernel_cpu_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, int numSrc, int nx, int ny, double mean)
{	
	for (int col=0; col < nx; ++col) {
		for (int row=0; row < ny; ++row) {
			double sum=0.0;
			double yr_x = col * dx + xmin;
			double yr_y = (ny - 1 - row) * dy + ymin;

			for (int i=0; i < numSrc; ++i) {
				switch (model) {
					case EXP:
						sum += covExpKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case GAU:
						sum += covGauKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case SPH:
						sum += covSphKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case MAT3:
						sum += covMat3Kernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					case MAT5:
						sum += covMat5Kernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget) * y[i]; 
						break;
					}

			}
			prediction[row * nx + col] = sum + mean; // TODO: TEST!
		}

	}
}


void krigingSimpleAnisKernel_cpu_2d(double *prediction, double2 *srcXY, double xmin, double dx, double ymin, double dy,  double *y,  int model, double range, double sill, double nugget, double alpha, double afac1, int numSrc, int nx, int ny, double mean)
{	
	for (int col=0; col < nx; ++col) {
		for (int row=0; row < ny; ++row) {
			double sum=0.0;
			double yr_x = col * dx + xmin;
			double yr_y = (ny - 1 - row) * dy + ymin;

			for (int i=0; i < numSrc; ++i) {
				switch (model) {
					case EXP:
						sum += covExpAnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case GAU:
						sum += covGauAnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case SPH:
						sum += covSphAnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case MAT3:
						sum += covMat3AnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					case MAT5:
						sum += covMat5AnisKernel_cpu_2d(yr_x,yr_y,srcXY[i].x,srcXY[i].y,sill,range,nugget,alpha,afac1) *y[i]; 
						break;
					}

			}
			prediction[row * nx + col] = sum + mean; // TODO: TEST!
		}

	}
}


void addResSim_cpu_2d(double *res, double *uncond, int n) 
{
	for (int i=0; i<n; ++i) 
		res[i] += uncond[i];
}

void addResSimMean_cpu_2d(double *res, double *uncond, int n, double mean) 
{
	for (int i=0; i<n; ++i) res[i] += uncond[i] + mean;
}



//Funktionen fuer approximiertes Kriging
void overlay_cpu_2d(double2 *out, double2 *xy, double grid_minx, double grid_dx, double grid_maxy, double grid_dy, int numPoints) 
{
	for (int i=0; i<numPoints; ++i) 
	{
		out[i].x = (xy[i].x - grid_minx)/grid_dx;
		out[i].y = (grid_maxy - xy[i].y)/grid_dy;
	}
}


//fuer jede Realisierung
void residualsOrdinary_cpu_2d(double* res, double *srcdata, double *uncond_grid, double2 *indices, int nx, int ny, int numPoints) {
	for (int id=0; id<numPoints; ++id) 
	{
		// Bilinear interpolation
		double x = indices[id].x; 
		double y = indices[id].y;
		int row = (int)floor(y); // y index of upper neighbour pixel
		int col = (int)floor(x); // x index of lower neighbour pixel
		x = (double)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (double)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour

		// Special case: last column / row
		if (col > nx-1) {
			x = 0.0;col = nx-1;
		}
		else if (col < 0) {
			x = 0.0;col=0;
		}
		if (row > nx-1) {
			y = 0.0;row = (int)(nx-y);
		}	
		else if (row < 0) {
			y = 0.0;row=0;
		}
		res[id] = srcdata[id] - ((1-y) * ((1-x) * uncond_grid[row * nx + col] + x * uncond_grid[row * nx + col + 1]) + 
								  y * ((1-x) * uncond_grid[(row+1) * nx + col] + x * uncond_grid[(row+1) * nx + col + 1]));
	}		

	res[numPoints] = 0.0; 
}





// Calculates residuals of samples and an unconditional realization. Uses bilinear interpolation based on the sample's position in grid
void residualsSimple_cpu_2d(double* res, double *srcdata, double *uncond_grid, double2 *indices, int nx, int ny, int numPoints, double mu) {
	for (int id=0; id<numPoints; ++id) {
		
		// Bilinear interpolation
		double x = indices[id].x; 
		double y = indices[id].y;
		int row = (int)floor(y); // y index of upper neighbour pixel
		int col = (int)floor(x); // x index of lower neighbour pixel
		x = (double)x - col; // Weight of right neighbour or 1 - weight of left neighbour
		y = (double)y - row; // Weight of lower neighbour or 1 - weight of upper neighbour

		// Special case: last column / row
		if (col > nx-1) {
			x = 0.0;col = nx-1;
		}
		else if (col < 0) {
			x = 0.0;col=0;
		}
		if (row > nx-1) {
			y = 0.0;row = (int)(nx-y);
		}	
		else if (row < 0) {
			y = 0.0;row=0;
		}
		res[id] = srcdata[id] - mu - ((1-y) * ((1-x) * uncond_grid[row * nx + col] + x * uncond_grid[row * nx + col + 1]) + 
								  y * ((1-x) * uncond_grid[(row+1) * nx + col] + x * uncond_grid[(row+1) * nx + col + 1]));
	}		
}











/*******************************************************************************************
** UNCONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/

// global variables for unconditional simulation. These data are needed in the preprocessing as well as in generating realizations
struct uncond_state_cpu_2d {
	fftw_complex *cov; // d_cov is the result of the preprocessing ans is needed for each realozation
	int nx,ny,n,m;
	double xmin,xmax,ymin,ymax,dx,dy;
	fftw_plan plan1;
	fftw_plan plan2;
	
} uncond_global_cpu_2d;


#ifdef __cplusplus
extern "C" {
#endif


void EXPORT unconditionalSimInit_cpu_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, 
									double *p_sill, double *p_range, double *p_nugget, int *p_covmodel, double *p_anis_direction, 
									double *p_anis_ratio, int *do_check, int *set_cov_to_zero, double *eigenvals_tol, int *ret_code) {
	*ret_code = OK;
	
	uncond_global_cpu_2d.nx= *p_nx; // Number of cols
	uncond_global_cpu_2d.ny= *p_ny; // Number of rows
     
	//Grid wird einfach verdoppelt, nicht auf naechst 2er-Potenz erweitert
	uncond_global_cpu_2d.n= 2*uncond_global_cpu_2d.nx; // Number of cols
	uncond_global_cpu_2d.m= 2*uncond_global_cpu_2d.ny; // Number of rows
	//uncond_global_cpu_2d.n = ceil2(2*uncond_global_cpu_2d.nx); /// 
	//uncond_global_cpu_2d.m = ceil2(2*uncond_global_cpu_2d.ny); /// 
	uncond_global_cpu_2d.dx = (*p_xmax - *p_xmin) / (uncond_global_cpu_2d.nx-1);
	uncond_global_cpu_2d.dy = (*p_ymax - *p_ymin) / (uncond_global_cpu_2d.ny-1);
	
	//cufftPlan2d(&uncond_global_cpu_2d.plan1, uncond_global_cpu_2d.n, uncond_global_cpu_2d.m, CUFFT_Z2Z); 
	

	fftw_complex *grid_c = fftw_alloc_complex(uncond_global_cpu_2d.m*uncond_global_cpu_2d.n);
	
	uncond_global_cpu_2d.plan1 = fftw_plan_dft_2d(uncond_global_cpu_2d.m, uncond_global_cpu_2d.n,grid_c,grid_c,FFTW_FORWARD,FFTW_ESTIMATE);
	uncond_global_cpu_2d.plan2 = fftw_plan_dft_2d(uncond_global_cpu_2d.m, uncond_global_cpu_2d.n,grid_c,grid_c,FFTW_BACKWARD,FFTW_ESTIMATE);
	
	
	// build grid (ROW MAJOR)
	for (int i=0; i<uncond_global_cpu_2d.n; ++i) { // i =  col index
		for (int j=0; j<uncond_global_cpu_2d.m; ++j) { // j = row index 
			grid_c[j*uncond_global_cpu_2d.n+i][0] = *p_xmin + i * uncond_global_cpu_2d.dx; 
			//h_grid_c[j*uncond_global_cpu_2d.n+i][1] = *p_ymin + (j+1) * uncond_global_cpu_2d.dy;  
			grid_c[j*uncond_global_cpu_2d.n+i][1] = *p_ymin + (uncond_global_cpu_2d.m-1-j)* uncond_global_cpu_2d.dy; 
			//h_grid_c[j*uncond_global_cpu_2d.n+i][1] = *p_ymax - j* uncond_global_cpu_2d.dy;

		}
	}
	
	
	double xc = *p_xmin + (uncond_global_cpu_2d.dx*uncond_global_cpu_2d.n)/2;
	double yc = *p_ymin +(uncond_global_cpu_2d.dy*uncond_global_cpu_2d.m)/2;
	double sill = *p_sill;
	double range = *p_range;
	double nugget = *p_nugget;
	bool isotropic = (*p_anis_ratio == 1.0);
	double afac1 = 1.0/(*p_anis_ratio);
	double alpha = (90.0 - *p_anis_direction) * (PI / 180.0);
	
	

	// Array for cov grid
	//cudaStatus = cudaMalloc((void**)&uncond_global_cpu_2d.d_cov,sizeof(fftw_complex)*uncond_global_cpu_2d.n*uncond_global_cpu_2d.m);
	uncond_global_cpu_2d.cov = fftw_alloc_complex(uncond_global_cpu_2d.m*uncond_global_cpu_2d.n);


	// Sample covariance and generate "trick" grid
	//cudaStatus = cudaMalloc((void**)&d_trick_grid_c,sizeof(fftw_complex)*uncond_global_cpu_2d.n*uncond_global_cpu_2d.m);
	fftw_complex *trick_grid_c = fftw_alloc_complex(uncond_global_cpu_2d.m*uncond_global_cpu_2d.n);;
	

	if (isotropic) {
		sampleCovKernel_cpu_2d(trick_grid_c, grid_c, uncond_global_cpu_2d.cov, xc, yc,*p_covmodel, sill, range,nugget,uncond_global_cpu_2d.n,uncond_global_cpu_2d.m);
		//cudaStatus = cudaThreadSynchronize();
		//if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sampleCovKernel_2d!\n", cudaStatus);	
	}
	else {	
		sampleCovAnisKernel_cpu_2d(trick_grid_c, grid_c, uncond_global_cpu_2d.cov, xc, yc, *p_covmodel, sill, range,nugget, alpha, afac1, uncond_global_cpu_2d.n,uncond_global_cpu_2d.m);	
		//cudaStatus = cudaThreadSynchronize();
		//if (cudaStatus != cudaSuccess)  printf("cudaThreadSynchronize returned error code %d after launching sampleCovAnisKernel_2d!\n", cudaStatus);	
	}


#ifdef DEBUG 
	{
		writeCSVMatrix("C:\\test\\grid_c.csv",grid_c,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		writeCSVMatrix("C:\\test\\trick_grid_c.csv",trick_grid_c,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		writeCSVMatrix("C:\\test\\cov.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
	}
#endif


	fftw_free(grid_c);
	//cudaFree(d_grid);






	// Execute 2d FFT of covariance grid in order to get the spectral representation 
	//cufftExecZ2Z(uncond_global_cpu_2d.plan1, uncond_global_cpu_2d.d_cov, uncond_global_cpu_2d.d_cov, CUFFT_FORWARD); // in place fft forward
	fftw_execute_dft(uncond_global_cpu_2d.plan1,uncond_global_cpu_2d.cov,uncond_global_cpu_2d.cov);


#ifdef DEBUG 
	{
		writeCSVMatrix("C:\\test\\fftcov.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
	}
#endif



	//cufftExecZ2Z(uncond_global_cpu_2d.plan1, d_trick_grid_c, d_trick_grid_c, CUFFT_FORWARD); // in place fft forward
	fftw_execute_dft(uncond_global_cpu_2d.plan1,trick_grid_c,trick_grid_c);

#ifdef DEBUG 
	{
		writeCSVMatrix("C:\\test\\ffttrickgrid.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
	}
#endif


	
	// Multiply fft of "trick" grid with n*m
	multKernel_cpu_2d(trick_grid_c, uncond_global_cpu_2d.n, uncond_global_cpu_2d.m);
	
#ifdef DEBUG 
	{
		writeCSVMatrix("C:\\test\\multffttrickgrid.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
	}
#endif

	// Devide spectral covariance grid by "trick" grid
	divideSpectrumKernel_cpu_2d(uncond_global_cpu_2d.cov, trick_grid_c,uncond_global_cpu_2d.n, uncond_global_cpu_2d.m,*eigenvals_tol);	
	
	
	#ifdef DEBUG 
	{
		writeCSVMatrix("C:\\test\\dividespectrum.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
	}
	#endif
	
	//cudaFree(d_trick_grid_c);
	fftw_free(trick_grid_c);


	if (*set_cov_to_zero) {
		setCov0Kernel_cpu_2d(uncond_global_cpu_2d.cov, uncond_global_cpu_2d.n*uncond_global_cpu_2d.m);	
	}


	if (*do_check) {
		//writeCSVMatrix("C:\\fft\\circulantm.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		for (int i=0; i<uncond_global_cpu_2d.n*uncond_global_cpu_2d.m; ++i) {
			if (uncond_global_cpu_2d.cov[i][0] < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				fftw_free(uncond_global_cpu_2d.cov);
				fftw_destroy_plan(uncond_global_cpu_2d.plan1);
				fftw_destroy_plan(uncond_global_cpu_2d.plan2);
				return;
			}	
		}
	}

	// Compute sqrt of cov grid
	sqrtKernel_cpu_2d(uncond_global_cpu_2d.cov,uncond_global_cpu_2d.n, uncond_global_cpu_2d.m);

	#ifdef DEBUG 
	{
		writeCSVMatrix("C:\\test\\sqrtfftcov.csv",uncond_global_cpu_2d.cov,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
	}
	#endif
	
}

// Generates unconditional realizations
// p_out = output array of size nx*ny*k * sizeof(double)
// p_k = Number of realizations
// ret_code = return code: 0=ok
void EXPORT unconditionalSimRealizations_cpu_2d(double *p_out,  int *p_k, int *ret_code)
{
	*ret_code = OK;
	
	int k = *p_k;

	double *rand = (double*)malloc(sizeof(double)*uncond_global_cpu_2d.n*uncond_global_cpu_2d.m);
	fftw_complex *fftrand = fftw_alloc_complex(uncond_global_cpu_2d.n * uncond_global_cpu_2d.m);
	fftw_complex* amp = fftw_alloc_complex(uncond_global_cpu_2d.n * uncond_global_cpu_2d.m);
	//double* out = (double*)malloc(sizeof(double)*uncond_global_cpu_2d.nx*uncond_global_cpu_2d.ny);

    //Realisierungen in Schleife, d.h. lineare Laufzeit in Bezug auf Realisierungen
	for(int l = 0; l<k; ++l) {
		
		// Generate Random Numbers
		//curandGenerateNormalDouble(curandGen,d_rand,uncond_global_cpu_2d.m*uncond_global_cpu_2d.n,0.0,1.0);
		randGenerateNormal(rand,uncond_global_cpu_2d.m*uncond_global_cpu_2d.n,0.0,1.0);

		// Convert real random numbers to complex numbers
		realToComplexKernel_cpu_2d(fftrand, rand, uncond_global_cpu_2d.n*uncond_global_cpu_2d.m);

#ifdef DEBUG 
		{
			std::stringstream ss;
			ss << "C:\\test\\" << l << "-rand.csv";
			writeCSVMatrix(ss.str().c_str()	,fftrand,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		}
#endif
		


		// Compute 2D FFT of random numbers
		//cufftExecZ2Z(uncond_global_cpu_2d.plan1, fftrand, fftrand, CUFFT_FORWARD); // in place fft forward
		fftw_execute_dft(uncond_global_cpu_2d.plan1,fftrand,fftrand);

#ifdef DEBUG 
		{
			std::stringstream ss;
			ss << "C:\\test\\" << l << "-fftrand.csv";
			writeCSVMatrix(ss.str().c_str()	,fftrand,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		}
#endif


		elementProduct_cpu_2d(amp, uncond_global_cpu_2d.cov, fftrand, uncond_global_cpu_2d.m*uncond_global_cpu_2d.n);  
		


#ifdef DEBUG 
		{
			std::stringstream ss;
			ss << "C:\\test\\" << l << "-amp.csv";
			writeCSVMatrix(ss.str().c_str()	,amp,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		}
#endif
		//cufftExecZ2Z(uncond_global_cpu_2d.plan1, d_amp, d_amp, CUFFT_INVERSE); // in place fft inverse for simulation		
		fftw_execute_dft(uncond_global_cpu_2d.plan2,amp,amp);

#ifdef DEBUG 
		{
			std::stringstream ss;
			ss << "C:\\test\\" << l << "-fftamp.csv";
			writeCSVMatrix(ss.str().c_str()	,amp,uncond_global_cpu_2d.m,uncond_global_cpu_2d.n);
		}
#endif
		ReDiv_cpu_2d(p_out + l*(uncond_global_cpu_2d.nx*uncond_global_cpu_2d.ny), amp, std::sqrt((double)(uncond_global_cpu_2d.n*uncond_global_cpu_2d.m)), uncond_global_cpu_2d.nx, uncond_global_cpu_2d.ny, uncond_global_cpu_2d.n);	
		//cudaMemcpy((p_out + l*(uncond_global_cpu_2d.nx*uncond_global_cpu_2d.ny)),d_out,sizeof(double)*uncond_global_cpu_2d.nx*uncond_global_cpu_2d.ny,cudaMemcpyDeviceToHost);
	}

	free(rand);
	fftw_free(fftrand);
	fftw_free(amp);

	//cudaFree(d_rand);
	//cudaFree(d_fftrand);
	//cudaFree(d_amp);
	//cudaFree(d_out);
	
}


void EXPORT unconditionalSimRelease_cpu_2d(int *ret_code) {
	*ret_code = OK;
	fftw_free(uncond_global_cpu_2d.cov);
	fftw_destroy_plan(uncond_global_cpu_2d.plan1);
	fftw_destroy_plan(uncond_global_cpu_2d.plan2);
}


#ifdef __cplusplus
}
#endif
















/*******************************************************************************************
** CONDITIONAL SIMULATION  ***************************************************************
********************************************************************************************/


// global variables for conditional simulation that are needed both, for initialization as well as for generating realizations
struct cond_state_cpu_2d {
	fftw_complex *cov; 
	int nx,ny,n,m;
	double xmin,xmax,ymin,ymax,dx,dy;
	double range, sill, nugget;
	double alpha, afac1;
	bool isotropic;
	
	fftw_plan plan1;
	fftw_plan plan2;

	// Variables for conditioning
	int numSrc; // Number of sample observation
	double2 *samplexy; // coordinates of samples
	double2 *sampleindices; // Corresponding grid indices in subpixel accuracy
	double *sampledata; // data values of samples
	//double *d_covinv; // inverse covariance matrix of samples
	double *uncond;
	//double *h_uncond; //cpu uncond cache
	int covmodel;
	int k;
	double mu; // known mean for simple kriging
	int krige_method;
	//double *residuals;
} cond_global_cpu_2d;













#ifdef __cplusplus
extern "C" {
#endif



void EXPORT conditionalSimInit_cpu_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, 
								  int *p_ny, double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, 
								  double *p_srcData, int *p_numSrc, int *p_covmodel, double *p_anis_direction, 
								  double *p_anis_ratio, int *do_check, int *set_cov_to_zero, double *eigenvals_tol, int *krige_method, double *mu, int *ret_code) {
	*ret_code = OK;

	cond_global_cpu_2d.nx= *p_nx; // Number of cols
	cond_global_cpu_2d.ny= *p_ny; // Number of rows
	cond_global_cpu_2d.n= 2*cond_global_cpu_2d.nx; // Number of cols
	cond_global_cpu_2d.m= 2*cond_global_cpu_2d.ny; // Number of rows
	cond_global_cpu_2d.dx = (*p_xmax - *p_xmin) / (cond_global_cpu_2d.nx - 1);
	cond_global_cpu_2d.dy = (*p_ymax - *p_ymin) / (cond_global_cpu_2d.ny - 1);
	cond_global_cpu_2d.numSrc = *p_numSrc;
	cond_global_cpu_2d.xmin = *p_xmin;
	cond_global_cpu_2d.xmax = *p_xmax;
	cond_global_cpu_2d.ymin = *p_ymin;
	cond_global_cpu_2d.ymax = *p_ymax;
	cond_global_cpu_2d.range = *p_range;
	cond_global_cpu_2d.sill = *p_sill;
	cond_global_cpu_2d.nugget = *p_nugget;
	cond_global_cpu_2d.covmodel = *p_covmodel;
	cond_global_cpu_2d.krige_method = *krige_method;
	if (cond_global_cpu_2d.krige_method == SIMPLE)
		cond_global_cpu_2d.mu = *mu;
	else cond_global_cpu_2d.mu = 0;
	cond_global_cpu_2d.isotropic = (*p_anis_ratio == 1.0);
	cond_global_cpu_2d.afac1 = 1.0/(*p_anis_ratio);
	cond_global_cpu_2d.alpha = (90.0 - *p_anis_direction) * (PI / 180.0);


	
	
	// Build grid (ROW MAJOR)
	fftw_complex *grid = fftw_alloc_complex(cond_global_cpu_2d.m*cond_global_cpu_2d.n);
	for (int i=0; i<cond_global_cpu_2d.n; ++i) { // i = col index
		for (int j=0; j<cond_global_cpu_2d.m; ++j) { // j = row index
			grid[j*cond_global_cpu_2d.n+i][0] = *p_xmin + i * cond_global_cpu_2d.dx; 
			//h_grid_c[j*cond_global_cpu_2d.n+i][1] = *p_ymin + (j+1) * cond_global_cpu_2d.dy;  
			grid[j*cond_global_cpu_2d.n+i][1] = *p_ymin + (cond_global_cpu_2d.m-1-j)* cond_global_cpu_2d.dy;
			//h_grid_c[j*cond_global_cpu_2d.n+i][1] = *p_ymax - j* cond_global_cpu_2d.dy;
		}
	}

	
	double xc = *p_xmin + (cond_global_cpu_2d.dx*cond_global_cpu_2d.n)/2;
	double yc = *p_ymin + (cond_global_cpu_2d.dy*cond_global_cpu_2d.m)/2;



	cond_global_cpu_2d.plan1 = fftw_plan_dft_2d(cond_global_cpu_2d.m, cond_global_cpu_2d.n,grid,grid,FFTW_FORWARD,FFTW_ESTIMATE);
	cond_global_cpu_2d.plan2 = fftw_plan_dft_2d(cond_global_cpu_2d.m, cond_global_cpu_2d.n,grid,grid,FFTW_BACKWARD,FFTW_ESTIMATE);

	
	// Sample covariance and generate "trick" grid
	cond_global_cpu_2d.cov = fftw_alloc_complex(cond_global_cpu_2d.m*cond_global_cpu_2d.n);
	fftw_complex *trick_grid_c = fftw_alloc_complex(cond_global_cpu_2d.m*cond_global_cpu_2d.n);

	if (cond_global_cpu_2d.isotropic) {
		sampleCovKernel_cpu_2d(trick_grid_c, grid, cond_global_cpu_2d.cov, xc, yc, cond_global_cpu_2d.covmodel, cond_global_cpu_2d.sill, cond_global_cpu_2d.range, cond_global_cpu_2d.nugget, cond_global_cpu_2d.n,cond_global_cpu_2d.m);
	}
	else {
		sampleCovAnisKernel_cpu_2d(trick_grid_c, grid, cond_global_cpu_2d.cov, xc, yc, cond_global_cpu_2d.covmodel, cond_global_cpu_2d.sill, cond_global_cpu_2d.range, cond_global_cpu_2d.nugget,cond_global_cpu_2d.alpha,cond_global_cpu_2d.afac1, cond_global_cpu_2d.n,cond_global_cpu_2d.m);
	}

	fftw_free(grid);


	// Compute spectral representation of cov and "trick" grid
	fftw_execute_dft(cond_global_cpu_2d.plan1,cond_global_cpu_2d.cov,cond_global_cpu_2d.cov);
	fftw_execute_dft(cond_global_cpu_2d.plan1,trick_grid_c,trick_grid_c);
	

	// Multiplication of fft(trick_grid) with n*m	
	multKernel_cpu_2d(trick_grid_c, cond_global_cpu_2d.n, cond_global_cpu_2d.m);
	

	// Devide spectral cov grid by fft of "trick" grid
	divideSpectrumKernel_cpu_2d(cond_global_cpu_2d.cov, trick_grid_c,cond_global_cpu_2d.n, cond_global_cpu_2d.m,*eigenvals_tol);	

	fftw_free(trick_grid_c);


	if (*set_cov_to_zero) {
		setCov0Kernel_cpu_2d(cond_global_cpu_2d.cov, cond_global_cpu_2d.n*cond_global_cpu_2d.m);	
	}



	// Copy to host and check for negative real parts
	if (*do_check) {
		//writeCSVMatrix("C:\\fft\\circulantm.csv",cond_global_cpu_2d.cov,cond_global_cpu_2d.m,cond_global_cpu_2d.n);	
		for (int i=0; i<cond_global_cpu_2d.n*cond_global_cpu_2d.m; ++i) {
			if (cond_global_cpu_2d.cov[i][0] < 0.0) {
				*ret_code = ERROR_NEGATIVE_COV_VALUES; 
				fftw_free(cond_global_cpu_2d.cov);
				fftw_destroy_plan(cond_global_cpu_2d.plan1);
				fftw_destroy_plan(cond_global_cpu_2d.plan2);
				return;
			}	
		}
	}

	// Compute sqrt of spectral cov grid
	sqrtKernel_cpu_2d(cond_global_cpu_2d.cov,cond_global_cpu_2d.n, cond_global_cpu_2d.m);
	

	cond_global_cpu_2d.samplexy = (double2*)malloc(sizeof(double2)* cond_global_cpu_2d.numSrc);
	cond_global_cpu_2d.sampleindices = (double2*)malloc(sizeof(double2)* cond_global_cpu_2d.numSrc);
	cond_global_cpu_2d.sampledata = (double*)malloc(sizeof(double)* cond_global_cpu_2d.numSrc);

	memcpy(cond_global_cpu_2d.samplexy,p_srcXY,sizeof(double2)* cond_global_cpu_2d.numSrc);
	memcpy(cond_global_cpu_2d.sampledata,p_srcData,sizeof(double)*cond_global_cpu_2d.numSrc);

	// Overlay samples to grid and save resulting subpixel grind indices
	overlay_cpu_2d(cond_global_cpu_2d.sampleindices,cond_global_cpu_2d.samplexy,*p_xmin,cond_global_cpu_2d.dx,*p_ymax,cond_global_cpu_2d.dy, cond_global_cpu_2d.numSrc);
	
}




// Generates Unconditional Realizations and the residuals of all samples to all realizations 
// p_out = output matrix of residuals, col means number of realization, row represents a sample point
// p_k = Number of realizations
// ret_code = return code: 0=ok


void EXPORT conditionalSimUncondResiduals_cpu_2d(double *p_out, int *p_k, int *ret_code) {
	*ret_code = OK;
	

	cond_global_cpu_2d.k = *p_k;
	double *rand = (double*)malloc(sizeof(double)*cond_global_cpu_2d.n*cond_global_cpu_2d.m);
	fftw_complex *fftrand = fftw_alloc_complex(cond_global_cpu_2d.n * cond_global_cpu_2d.m);
	fftw_complex* amp = fftw_alloc_complex(cond_global_cpu_2d.n * cond_global_cpu_2d.m);
	
	cond_global_cpu_2d.uncond = (double*)malloc(sizeof(double)*cond_global_cpu_2d.nx*cond_global_cpu_2d.ny * cond_global_cpu_2d.k);
	/*if (cond_global_cpu_2d.krige_method == ORDINARY) {
		cond_global_cpu_2d.residuals = (double*)malloc(sizeof(double)*  (cond_global_cpu_2d.numSrc + 1));
	}
	else if (cond_global_cpu_2d.krige_method == SIMPLE) {
		cond_global_cpu_2d.residuals = (double*)malloc(sizeof(double)*  (cond_global_cpu_2d.numSrc));
	}*/
		
	
		
	for(int l=0; l<cond_global_cpu_2d.k; ++l) {
			
		randGenerateNormal(rand,cond_global_cpu_2d.m*cond_global_cpu_2d.n,0.0,1.0);
		realToComplexKernel_cpu_2d(fftrand, rand, cond_global_cpu_2d.n*cond_global_cpu_2d.m);
		fftw_execute_dft(cond_global_cpu_2d.plan1,fftrand,fftrand);
		elementProduct_cpu_2d(amp, cond_global_cpu_2d.cov, fftrand, cond_global_cpu_2d.m*cond_global_cpu_2d.n);
		fftw_execute_dft(cond_global_cpu_2d.plan2,amp,amp);
		ReDiv_cpu_2d(cond_global_cpu_2d.uncond + l*cond_global_cpu_2d.nx*cond_global_cpu_2d.ny, amp, std::sqrt((double)(cond_global_cpu_2d.n*cond_global_cpu_2d.m)), cond_global_cpu_2d.nx, cond_global_cpu_2d.ny, cond_global_cpu_2d.n);

		// uncond is now a unconditional realization 
		// Compute residuals between samples and uncond
		
		if (cond_global_cpu_2d.krige_method == ORDINARY) {
			residualsOrdinary_cpu_2d(p_out + l*(cond_global_cpu_2d.numSrc + 1),cond_global_cpu_2d.sampledata,cond_global_cpu_2d.uncond+l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny),cond_global_cpu_2d.sampleindices,cond_global_cpu_2d.nx,cond_global_cpu_2d.ny,cond_global_cpu_2d.numSrc);
		}
		else if (cond_global_cpu_2d.krige_method == SIMPLE) {
			residualsSimple_cpu_2d(p_out + l*cond_global_cpu_2d.numSrc,cond_global_cpu_2d.sampledata,cond_global_cpu_2d.uncond+l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny),cond_global_cpu_2d.sampleindices,cond_global_cpu_2d.nx,cond_global_cpu_2d.ny,cond_global_cpu_2d.numSrc, cond_global_cpu_2d.mu);
		}
		
		

		//if (cond_global_cpu_2d.krige_method == ORDINARY) {
		//	//cudaMemcpy((p_out + l*(cond_global_cpu_2d.numSrc + 1)),cond_global_cpu_2d.d_residuals,sizeof(double)* (cond_global_cpu_2d.numSrc + 1),cudaMemcpyDeviceToHost);	
		//}
		//else if (cond_global_cpu_2d.krige_method == SIMPLE) {
		//	//cudaMemcpy(p_out + l*cond_global_cpu_2d.numSrc,cond_global_cpu_2d.d_residuals,sizeof(double) * cond_global_cpu_2d.numSrc,cudaMemcpyDeviceToHost);	
		//}
		
	}
	
	free(rand);
	fftw_free(fftrand);
	fftw_free(amp);

}


void EXPORT conditionalSimKrigeResiduals_cpu_2d(double *p_out, double *p_y, int *ret_code)
{
	*ret_code = OK;
	for(int l = 0; l<cond_global_cpu_2d.k; ++l) {					
		if (cond_global_cpu_2d.isotropic) {
			krigingKernel_cpu_2d(p_out + l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny),cond_global_cpu_2d.samplexy,cond_global_cpu_2d.xmin,cond_global_cpu_2d.dx,cond_global_cpu_2d.ymin,cond_global_cpu_2d.dy,p_y + l*(cond_global_cpu_2d.numSrc + 1),cond_global_cpu_2d.covmodel,cond_global_cpu_2d.range,cond_global_cpu_2d.sill,cond_global_cpu_2d.nugget,cond_global_cpu_2d.numSrc,cond_global_cpu_2d.nx,cond_global_cpu_2d.ny);
		}
		else 	{
			krigingAnisKernel_cpu_2d(p_out + l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny),cond_global_cpu_2d.samplexy,cond_global_cpu_2d.xmin,cond_global_cpu_2d.dx,cond_global_cpu_2d.ymin,cond_global_cpu_2d.dy,p_y + l*(cond_global_cpu_2d.numSrc + 1),cond_global_cpu_2d.covmodel,cond_global_cpu_2d.range,cond_global_cpu_2d.sill,cond_global_cpu_2d.nugget,cond_global_cpu_2d.alpha,cond_global_cpu_2d.afac1,cond_global_cpu_2d.numSrc,cond_global_cpu_2d.nx,cond_global_cpu_2d.ny);
		}	
		addResSim_cpu_2d(p_out + l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny), cond_global_cpu_2d.uncond + l*cond_global_cpu_2d.nx*cond_global_cpu_2d.ny, cond_global_cpu_2d.nx*cond_global_cpu_2d.ny);
	}
}




void EXPORT conditionalSimSimpleKrigeResiduals_cpu_2d(double *p_out, double *p_y, int *ret_code)
{
	*ret_code = OK;
	for(int l = 0; l<cond_global_cpu_2d.k; ++l) {				
		if (cond_global_cpu_2d.isotropic) {
			krigingSimpleKernel_cpu_2d(p_out + l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny),cond_global_cpu_2d.samplexy,cond_global_cpu_2d.xmin,cond_global_cpu_2d.dx,cond_global_cpu_2d.ymin,cond_global_cpu_2d.dy,p_y + l*(cond_global_cpu_2d.numSrc),cond_global_cpu_2d.covmodel,cond_global_cpu_2d.range,cond_global_cpu_2d.sill,cond_global_cpu_2d.nugget,cond_global_cpu_2d.numSrc,cond_global_cpu_2d.nx,cond_global_cpu_2d.ny,cond_global_cpu_2d.mu);
		}
		else 	{
			krigingSimpleAnisKernel_cpu_2d(p_out + l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny),cond_global_cpu_2d.samplexy,cond_global_cpu_2d.xmin,cond_global_cpu_2d.dx,cond_global_cpu_2d.ymin,cond_global_cpu_2d.dy,p_y + l*(cond_global_cpu_2d.numSrc),cond_global_cpu_2d.covmodel,cond_global_cpu_2d.range,cond_global_cpu_2d.sill,cond_global_cpu_2d.nugget,cond_global_cpu_2d.alpha,cond_global_cpu_2d.afac1,cond_global_cpu_2d.numSrc,cond_global_cpu_2d.nx,cond_global_cpu_2d.ny,cond_global_cpu_2d.mu);
		}
		addResSim_cpu_2d(p_out + l*(cond_global_cpu_2d.nx*cond_global_cpu_2d.ny), cond_global_cpu_2d.uncond + l*cond_global_cpu_2d.nx*cond_global_cpu_2d.ny, cond_global_cpu_2d.nx*cond_global_cpu_2d.ny);	
	}
}





void EXPORT conditionalSimRelease_cpu_2d(int *ret_code) {
	*ret_code = OK;
	fftw_destroy_plan(cond_global_cpu_2d.plan1);
	fftw_destroy_plan(cond_global_cpu_2d.plan2);
	free(cond_global_cpu_2d.samplexy);
	free(cond_global_cpu_2d.sampledata);
	free(cond_global_cpu_2d.sampleindices);
	fftw_free(cond_global_cpu_2d.cov);
	free(cond_global_cpu_2d.uncond);
}




#ifdef __cplusplus
}
#endif


