
#include <stdio.h>
#include <stdlib.h>

#include <cufft.h>
#include <cuda_runtime.h>




#define MAX_PLANS 10



typedef struct {
  int nx;
  int ny;
  int nz;
  int use_double;
  cufftHandle cufftPlan;
} fft_plan;



fft_plan *plans[MAX_PLANS];
int cur_plan_index;



#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// C oder C++ Compiler
#ifdef __cplusplus
extern "C" {
#endif

//alle Funktionen, die mit EXPORT gekennzeichnet sind, koennen in R ueber .C aufgerufen werden
void EXPORT initFFT() {

	for (int i=0; i<MAX_PLANS; ++i) {
		plans[i] = 0; // init plans with null pointers
	}
	cur_plan_index = 0;
}



void EXPORT planFFT(int *nx, int *ny, int *nz, int* use_double) {
  fft_plan *plan = (fft_plan*)malloc(sizeof(fft_plan));
  plan->nx = *nx; //*plan.nx = nx;
  plan->ny = *ny;
  plan->nz = *nz;
  plan->use_double = *use_double;

  //plan->h_data  = (cufftComplex*)malloc(sizeof(cufftComplex) * plan->nx * plan->ny * plan->nz);
  //cudaMalloc((void**)&plan->d_data,sizeof(cufftComplex) * plan->nx * plan->ny * plan->nz);
  
  cufftType type = (*use_double)? CUFFT_Z2Z : CUFFT_C2C;
  // if (*use_double) type=CUFFT_Z2Z else type=CUFFT_C2C

  // Plan je nach Dimension
  if (plan->nz == 1 && plan->ny == 1) {
	cufftPlan1d(&plan->cufftPlan, plan->nx, type, 1);
  }
  else if (plan->nz == 1) {
	cufftPlan2d(&plan->cufftPlan, plan->ny, plan->nx, type); // cufft uses row-major
  }
  else {
	cufftPlan3d(&plan->cufftPlan, plan->nz, plan->ny, plan->nx, type); // cufft uses row-major
  }
  
  if (plans[cur_plan_index] != 0) {
	  cufftDestroy(plans[cur_plan_index]->cufftPlan);
	  free(plans[cur_plan_index]);
	  plans[cur_plan_index] = 0;
  }
  plans[cur_plan_index] = plan;
  cur_plan_index = (cur_plan_index + 1) % MAX_PLANS;
}




void EXPORT execFFT(double *out, double *data, int *nx, int *ny, int *nz, int *inverse, int *is_complex, int *use_double) {

	// Is there already a plan?!
	fft_plan *plan = 0;
	for (int i=0; i<MAX_PLANS; ++i) {
		if (plans[i] != 0) {
			if (plans[i]->nx == *nx && plans[i]->ny == *ny && plans[i]->nz == *nz && plans[i]->use_double == *use_double) {
				plan = plans[i];
				break;
			}
		}
	}
	// If not, create one
	if (plan == 0) {
		planFFT(nx,ny,nz,use_double);
		plan = plans[(cur_plan_index - 1) % MAX_PLANS];
	}


	int direction = CUFFT_FORWARD;
	if (*inverse) direction = CUFFT_INVERSE;  
  
	int n = plan->nx * plan->ny * plan->nz;
	
	
	if (*use_double) {
		cufftDoubleComplex *h_data = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * n);
		cufftDoubleComplex *d_data;
		cudaMalloc((void**)&d_data,sizeof(cufftDoubleComplex) * n);

		if (!*is_complex) {
			for (int i = 0; i < n; ++i) {
			  h_data[i].x = (double)data[i];
			  h_data[i].y = 0.0;
			}
		}
		else {
			for (int i = 0; i < n; ++i) {
			  h_data[i].x = (double)data[2*i];
			  h_data[i].y = (double)data[2*i+1];
			}

		}
		// Copy data to GPU memory
		cudaMemcpy(d_data,h_data,sizeof(cufftDoubleComplex) * n, cudaMemcpyHostToDevice);
		// Execute FFT in place
		cufftExecZ2Z(plan->cufftPlan,d_data,d_data,direction);
		// Copy data back to host memory
		cudaMemcpy(h_data,d_data,sizeof(cufftDoubleComplex) * n, cudaMemcpyDeviceToHost);	

        // komplexer Datentyp von CUDA wird in 2*n Array umgewandelt
		for (int i = 0; i < n; ++i) {
			out[2*i] = h_data[i].x;
			out[2*i+1] = h_data[i].y;
		}
		free(h_data);
		cudaFree(d_data);
	}
    // das gleiche fuer float
	else {
		cufftComplex *h_data = (cufftComplex*)malloc(sizeof(cufftComplex) * n);
		cufftComplex *d_data;
		cudaMalloc((void**)&d_data,sizeof(cufftComplex) * n);

		if (!*is_complex) {
			for (int i = 0; i < n; ++i) {
			  h_data[i].x = (float)data[i];
			  h_data[i].y = 0.0f;
			}
		}
		else {
			for (int i = 0; i < n; ++i) {
			  h_data[i].x = (float)data[2*i];
			  h_data[i].y = (float)data[2*i+1];
			}

		}
		// Copy data to GPU memory
		cudaMemcpy(d_data,h_data,sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);
		// Execute FFT in place
		cufftExecC2C(plan->cufftPlan,d_data,d_data,direction);
		// Copy data back to host memory
		cudaMemcpy(h_data,d_data,sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost);	


		for (int i = 0; i < n; ++i) {
			out[2*i] = h_data[i].x;
			out[2*i+1] = h_data[i].y;
		}
		free(h_data);
		cudaFree(d_data);
	}
}






#ifdef __cplusplus
}
#endif
