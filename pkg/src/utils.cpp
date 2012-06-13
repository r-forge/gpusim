#include "utils.h"


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

void writeCSVMatrix(const char *filename, cufftComplex* matrix, int numRows, int numCols) {
	using namespace std;
	
	fstream file;
	file.open(filename, ios::out);
	if (file.fail()) return;

	for (int i=0; i<numRows; ++i) {
		for (int j = 0; j<numCols; ++j) {
			file << matrix[i * numCols + j].x << " ";
		}
		file << "\n";
	}
	file << "\n";file << "\n";
	for (int i=0; i<numRows; ++i) {
		for (int j = 0; j<numCols; ++j) {
			file << matrix[i * numCols + j].y << " ";
		}
		file << "\n";
	}
	file.close();
}



int ceil2(int n) {
	int out = 1;
	while(out<n) out*=2;
	return out;
}




#ifdef __cplusplus
extern "C" {
#endif



void EXPORT initSim(int *result) {
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)  {
			printf("cudaSetDevice returned error code %d\n", cudaStatus);
			*result = ERROR_NO_DEVICE;
		}
		*result = OK;
	}

	void EXPORT deviceInfo(char **info) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props,0);
		sprintf(info[0],"Device: %i - %s\nCUDA Version: %i.%i\nMultiprocessors: %i\nClock Frequency: %iMHz\nGlobal Memory: %iMiB\nShared Memory: %iKiB\nRegisters per Block: %i\nConstant Memory: %iKiB\n",
		0,props.name,props.major,props.minor,props.multiProcessorCount,(int)(props.clockRate / 1000),(int)(props.totalGlobalMem/(1024*1024)),(int)(props.sharedMemPerBlock/(1024)),props.regsPerBlock,(int)(props.totalConstMem/1024));
	}



double inline covExp_scalar(double d, double sill, double range, double nugget) {
	return ((d == 0.0)? (nugget + sill) : (sill*exp(-d/(range))));
}

double inline covGau_scalar(double d, double sill, double range, double nugget) {
	return ((d == 0.0)? (nugget + sill) : (sill*exp(-d*d) / (range*range) ));
}
double inline covSph_scalar(double d, double sill, double range, double nugget) {
	if (d == 0.0) return (nugget + sill);	
	else if(d <= range) return sill * (1.0 - (((3*d) / (2*range)) - ((d*d*d) / (2*range*range*range)) ));	
	else return 0.0; 
}
double inline covMat3_scalar(double d, double sill, double range, double nugget) {
	return ((d == 0.0)? (nugget + sill) : (sill*(1+SQRT3*d/range)*exp(-SQRT3*d/range)));
}
double inline covMat5_scalar(double d, double sill, double range, double nugget) {
	return ((d == 0.0)? (nugget + sill) : (sill * (1 + SQRT5*d/range + 5*d*d/3*range*range) * exp(-SQRT5*d/range)));
}



// Covariance functions
void EXPORT covExp(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		out[i] = covExp_scalar(h[i],*sill,*range,*nugget);
	}
}

void EXPORT covGau(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		out[i] = covGau_scalar(h[i],*sill,*range,*nugget);
	}
}
void EXPORT covSph(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		out[i] = covSph_scalar(h[i],*sill,*range,*nugget);
	}
}
void EXPORT covMat3(double *out, double *h, int *n, double *sill, double *range, double *nugget) {		
	for (int i=0; i<*n; ++i) {		
		out[i] = covMat3_scalar(h[i],*sill,*range,*nugget);
	}
}
void EXPORT covMat5(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {
		out[i] = covMat5_scalar(h[i],*sill,*range,*nugget);
	}
}






///// NEW INTERFACE FUNCTIONS FOR ALL COV FUNCS AND 2 AS WELL AS 3 DIMS


void EXPORT dCov(double *out, double *xy, int *n, int *dims, int *model, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = 0.0;
			for (int d=0; d<*dims; ++d) {
				dist += (xy[i+(d*(*n))] - xy[j+(d*(*n))]) * (xy[i+(d*(*n))] - xy[j+(d*(*n))]);				
			}
			dist = sqrt(dist);
			// Compute cov
			switch(*model) {
			case EXP:
				dist = covExp_scalar(dist,*sill,*range,*nugget);
				break;
			case GAU:
				dist = covGau_scalar(dist,*sill,*range,*nugget);
				break;
			case SPH:
				dist = covSph_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT3:
				dist = covMat3_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT5:
				dist = covMat5_scalar(dist,*sill,*range,*nugget);
				break;
			}
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	


}


void EXPORT dCovAnis3d(double *out, double *xy, int *n, int *model, double *sill, double *range, double *nugget, double *anis_dir1, double *anis_dir2, double *anis_dir3, double *anis_rat1, double *anis_rat2) {
	double alpha = (90.0 - *anis_dir1) * (PI / 180.0);
	double beta = -1.0 * *anis_dir2 * (PI / 180.0);
	double theta = *anis_dir2 * (PI / 180.0);
	double afac1 = 1/(*anis_rat1);
	double afac2 = 1/(*anis_rat2);


	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {	

			double dx = xy[i] - xy[j];
			double dy = xy[i+*n] - xy[j+*n];
			double dz = xy[i+2*(*n)] - xy[j+2*(*n)];
			
			double dist = 0.0;
			double temp = 0.0;

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

			switch(*model) {
			case EXP:
				dist = covExp_scalar(dist,*sill,*range,*nugget);
				break;
			case GAU:
				dist = covGau_scalar(dist,*sill,*range,*nugget);
				break;
			case SPH:
				dist = covSph_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT3:
				dist = covMat3_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT5:
				dist = covMat5_scalar(dist,*sill,*range,*nugget);
				break;
			}			
			
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	
}



void EXPORT dCov3d(double *out, double *xy, int *n, int *model, double *sill, double *range, double *nugget) {
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {	

			double dx = xy[i] - xy[j];
			double dy = xy[i+*n] - xy[j+*n];
			double dz = xy[i+2*(*n)] - xy[j+2*(*n)];
			double dist = sqrt(dx*dx+dy*dy+dz*dz);

			switch(*model) {
			case EXP:
				dist = covExp_scalar(dist,*sill,*range,*nugget);
				break;
			case GAU:
				dist = covGau_scalar(dist,*sill,*range,*nugget);
				break;
			case SPH:
				dist = covSph_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT3:
				dist = covMat3_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT5:
				dist = covMat5_scalar(dist,*sill,*range,*nugget);
				break;
			}			
			
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	
}









// Distance and covariance matrix calculation, input coordinates are NOT interleaved, this fits better to an R data.frame
void EXPORT dCovAnis2d(double *out, double *xy, int *n, int *model, double *sill, double *range, double *nugget, double *anis_majordir, double *anis_ratio) {	
	double alpha = (90.0 - *anis_majordir) * (PI / 180.0);
	double afac1 = 1.0/(*anis_ratio);
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {	

			double dx = xy[i] - xy[j];
			double dy = xy[i+*n] - xy[j+*n];
			
			double dist = 0.0;
			double temp = 0.0;

			temp = dx * cos(alpha) + dy * sin(alpha);
			dist += temp * temp;

			temp = afac1 * (dx * (-sin(alpha)) + dy * cos(alpha));
			dist += temp * temp;

			dist = sqrt(dist);


			switch(*model) {
			case EXP:
				dist = covExp_scalar(dist,*sill,*range,*nugget);
				break;
			case GAU:
				dist = covGau_scalar(dist,*sill,*range,*nugget);
				break;
			case SPH:
				dist = covSph_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT3:
				dist = covMat3_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT5:
				dist = covMat5_scalar(dist,*sill,*range,*nugget);
				break;
			}			
			
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	
}



// Distance and covariance matrix calculation, input coordinates are NOT interleaved, this fits better to an R data.frame
void EXPORT dCov2d(double *out, double *xy, int *n, int *model, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {	

			double dx = xy[i] - xy[j];
			double dy = xy[i+*n] - xy[j+*n];		
			double dist = sqrt(dx*dx+dy*dy);
			switch(*model) {
			case EXP:
				dist = covExp_scalar(dist,*sill,*range,*nugget);
				break;
			case GAU:
				dist = covGau_scalar(dist,*sill,*range,*nugget);
				break;
			case SPH:
				dist = covSph_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT3:
				dist = covMat3_scalar(dist,*sill,*range,*nugget);
				break;
			case MAT5:
				dist = covMat5_scalar(dist,*sill,*range,*nugget);
				break;
			}					
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	
}








#ifdef __cplusplus
}
#endif