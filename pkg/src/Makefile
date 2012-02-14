OBJ_FILES = sim.o fft.o
DLLOUT = gpusim.so

CUDA_HOME = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_HOME)/include
CUDA_LIB_PATH = $(CUDA_HOME)/lib64
NVCC = $(CUDA_HOME)/bin/nvcc

LIBS = -lcuda -lcudart -lcublas -lcurand -lcufft

	
NVCC_FLAGS = -gencode arch=compute_10,code=sm_10 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -Xcompiler -fPIC


all: $(OBJ_FILES)
	$(NVCC) -shared -L$(CUDA_LIB_PATH) $(LIBS) $(OBJ_FILES) -o $(DLLOUT) 

sim.o: sim.cu
	$(NVCC) -c sim.cu -o sim.o -I$(CUDA_INC_PATH) $(NVCC_FLAGS)
	
sim.o: fft.cu
	$(NVCC) -c fft.cu -o fft.o -I$(CUDA_INC_PATH) $(NVCC_FLAGS)

clean:
	rm $(OBJ_FILES) gpusim.so
