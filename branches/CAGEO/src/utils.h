#include "Rinterface.h"


#define SQRT3 1.732050807568877
#define SQRT5 2.23606797749979



int ceil2(int n);
void writeCSVMatrix(const char *filename, cufftComplex* matrix, int numRows, int numCols);
void writeCSVMatrix(const char *filename, float* matrix, int numRows, int numCols);