
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

#ifdef PI
#undef PI
#endif
#define PI 3.1415926535897932384626433832795 
enum covfunc {EXP=0, SPH=1, GAU=2, MAT3=3, MAT5=4}; 
enum kriging {SIMPLE=0, ORDINARY=1};
enum errcode {OK=0, ERROR_NEGATIVE_COV_VALUES=1, ERROR_UNKNOWN = 2, ERROR_NO_DEVICE = 3};



//#define LOG



/*

#ifdef __cplusplus
extern "C" {
#endif

	void EXPORT initSim(int *result);
	void EXPORT deviceInfo(char **info);

	void EXPORT covExp(double *out, double *h, int *n, double *sill, double *range, double *nugget);
	void EXPORT covGau(double *out, double *h, int *n, double *sill, double *range, double *nugget);
	void EXPORT covSph(double *out, double *h, int *n, double *sill, double *range, double *nugget);

	void EXPORT dCovExp2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget);
	void EXPORT dCovExp3d(double *out, double *xy, int *n, double *sill, double *range, double *nugget);
	void EXPORT dCovGau2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget);
	void EXPORT dCovGau3d(double *out, double *xy, int *n, double *sill, double *range, double *nugget);
	void EXPORT dCovSph2d(double *out, double *xy, int *n, double *sill, double *range, double *nugget);
	void EXPORT dCovSph3d(double *out, double *xy, int *n, double *sill, double *range, double *nugget);




	void EXPORT unconditionalSimInit_2f(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, double *p_sill, double *p_range, double *p_nugget, int *p_covmodel, int *do_check, int *ret_code);
	void EXPORT unconditionalSimRealizations_2f(double *p_out,  int *p_k, int *ret_code);
	void EXPORT unconditionalSimRelease_2f(int *ret_code);

	void EXPORT conditionalSimInit_2f(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, double *p_srcData, int *p_numSrc, int *p_covmodel, int *do_check, int *krige_method, double *mu, int *ret_code);
	void EXPORT conditionalSimUncondResiduals_2f(double *p_out, int *p_k, int *ret_code);
	void EXPORT conditionalSimKrigeResiduals_2f(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditionalSimSimpleKrigeResiduals_2f(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditionalSimRelease_2f(int *ret_code);

	void EXPORT conditioningInit_2f(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, double *p_srcData, int *p_numSrc,  int *p_k, double *p_uncond, int *p_covmodel, int *krige_method, double *mu, int *ret_code);
	void EXPORT conditioningResiduals_2f(double *p_out, int *ret_code);
	void EXPORT conditioningKrigeResiduals_2f(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditioningSimpleKrigeResiduals_2f(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditioningRelease_2f(int *ret_code);





	void EXPORT unconditionalSimInit_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, double *p_sill, double *p_range, double *p_nugget, int *p_covmodel, int *do_check, int *ret_code);
	void EXPORT unconditionalSimRealizations_2d(double *p_out,  int *p_k, int *ret_code);
	void EXPORT unconditionalSimRelease_2d(int *ret_code);

	void EXPORT conditionalSimInit_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, double *p_srcData, int *p_numSrc, int *p_covmodel, int *do_check, int *krige_method, double *mu, int *ret_code);
	void EXPORT conditionalSimUncondResiduals_2d(double *p_out, int *p_k, int *ret_code);
	void EXPORT conditionalSimKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditionalSimSimpleKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditionalSimRelease_2d(int *ret_code);

	void EXPORT conditioningInit_2d(double *p_xmin, double *p_xmax, int *p_nx, double *p_ymin, double *p_ymax, int *p_ny, double *p_sill, double *p_range, double *p_nugget, double *p_srcXY, double *p_srcData, int *p_numSrc,  int *p_k, double *p_uncond, int *p_covmodel, int *krige_method, double *mu, int *ret_code);
	void EXPORT conditioningResiduals_2d(double *p_out, int *ret_code);
	void EXPORT conditioningKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditioningSimpleKrigeResiduals_2d(double *p_out, double *p_y, int *ret_code);
	void EXPORT conditioningRelease_2d(int *ret_code);



#ifdef __cplusplus
}
#endif


*/