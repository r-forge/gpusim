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


void EXPORT covMat3(double *out, double *h, int *n, double *sill, double *range, double *nugget) {	
	float SQRT3 = 1.732050807568877;	
	for (int i=0; i<*n; ++i) {		
		out[i] = ((h[i] == 0.0)? (*nugget + *sill) : (*sill*(1+SQRT3*h[i]/(*range))*exp(-SQRT3*h[i]/(*range))));
	}
}


// TODO: Range?
void EXPORT covMat5(double *out, double *h, int *n, double *sill, double *range, double *nugget) {
	double SQRT5 = 2.23606797749979;
	for (int i=0; i<*n; ++i) {
		out[i] = ((h[i] == 0.0)? (*nugget + *sill) : (*sill * (1 + SQRT5*h[i]/(*range) + 5*h[i]*h[i]/3*(*range)*(*range)) * exp(-SQRT5*h[i]/(*range))));
	}
}



// Distance and covariance matrix calculation, input coordinates are NOT interleaved, this fits better to an R data.frame
void EXPORT dCovExp_2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = sqrt((xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]));
			// Compute cov
			dist = ((dist == 0.0)? (*nugget + *sill) : (*sill*exp(-dist/(*range)))); // Or assume no two points having the same coords?!
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	
}

// Distance and covariance matrix calculation, input coordinates are NOT interleaved, this fits better to an R data.frame
void EXPORT dCovExp_3d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = sqrt((xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]) + (xy[i+2*(*n)] - xy[j+2*(*n)]) * (xy[i+2*(*n)] - xy[j+2*(*n)]));
			// Compute cov
			dist = ((dist == 0.0)? (*nugget + *sill) : (*sill*exp(-dist/(*range)))); // Or assume no two points having the same coords?!
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;	
		}
	}	
}


void EXPORT dCovGau_2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist2 = (xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]);
			// Compute cov
			dist2 = ((dist2 == 0.0)? (*nugget + *sill) : (*sill*exp(- (dist2) / ((*range)*(*range))  ))); // Or assume no two points having the same coords?!
			out[i*(*n)+j] = dist2;
			out[j*(*n)+i] = dist2;
		}
	}
}
void EXPORT dCovGau_3d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist2 = (xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]) + (xy[i+2*(*n)] - xy[j+2*(*n)]) * (xy[i+2*(*n)] - xy[j+2*(*n)]);
			// Compute cov
			dist2 = ((dist2 == 0.0)? (*nugget + *sill) : (*sill*exp(- (dist2) / ((*range)*(*range))  ))); // Or assume no two points having the same coords?!
			out[i*(*n)+j] = dist2;
			out[j*(*n)+i] = dist2;
		}
	}
}



void EXPORT dCovSph_2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = sqrt((xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]));
			// Compute cov
			if (dist == 0.0) dist = (*nugget + *sill);		
			else if (dist <= *range) dist = *sill * (1.0 - (((3.0*dist) / (2.0*(*range))) - ((dist * dist * dist) / (2.0 * (*range) * (*range) * (*range))) ));						
			else dist = 0.0;		
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;
		}
	}	
}


void EXPORT dCovSph_3d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = sqrt((xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]) + (xy[i+2*(*n)] - xy[j+2*(*n)]) * (xy[i+2*(*n)] - xy[j+2*(*n)]));
			// Compute cov
			if (dist == 0.0) dist = (*nugget + *sill);		
			else if (dist <= *range) dist = *sill * (1.0 - (((3.0*dist) / (2.0*(*range))) - ((dist * dist * dist) / (2.0 * (*range) * (*range) * (*range))) ));						
			else dist = 0.0;		
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;
		}
	}
}










void EXPORT dCovMat3_2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	double SQRT3 = 1.732050807568877;
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = sqrt((xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]));
			dist = ((dist == 0.0)? (*nugget + *sill) : (*sill*(1+SQRT3*dist/(*range))*exp(-SQRT3*dist/(*range))));
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;
		}
	}
}



void EXPORT dCovMat5_2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget) {	
	double SQRT5 = 2.23606797749979;
	for (int i=0; i<*n; ++i) {	
		// Diagonal elements here
		out[i*(*n)+i] = (*nugget + *sill);
		for (int j=i+1; j<*n; ++j) {			
			// Compute distance
			double dist = sqrt((xy[i] - xy[j]) * (xy[i] - xy[j]) + (xy[i+*n] - xy[j+*n]) * (xy[i+*n] - xy[j+*n]));		
			dist = ((dist == 0.0)? (*nugget + *sill) : (*sill * (1 + SQRT5*dist/(*range) + 5*dist*dist/3*(*range)*(*range)) * exp(-SQRT5*dist/(*range))));
			out[i*(*n)+j] = dist;
			out[j*(*n)+i] = dist;
		}
	}
}





#ifdef __cplusplus
}
#endif