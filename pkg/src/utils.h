#include "Rinterface.h"

int ceil2(int n);
void writeCSVMatrix(const char *filename, cufftComplex* matrix, int numRows, int numCols);
void writeCSVMatrix(const char *filename, float* matrix, int numRows, int numCols);