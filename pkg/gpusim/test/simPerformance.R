library(gpusim)
library(RandomFields)
library(fields)




# This function checks the performance of unconditional simulation for different input configurations and compares the measured computation times to the fields package
simPerformance <- function() {


ITERATIONS = 2 # How often are computations repeated? Computation times will be averaged
NUM_REALIZATIONS = c(10,50,100,250,500) ## Which numbers of realizations should be tested
GRID_DIMS = c(100,200,300,400,500) ### Welche grid sized should be tested, note that the number of cells is squared to these numbers

range = 2
sill = 5
nugget = 0
xmin = 0
xmax = 5
ymin = 0
ymax = 5
	

## Result as matrices
## Cols: Numbers of realizations
## Rows: Grid sizes
outGPU = matrix(rep(0,length(NUM_REALIZATIONS)*length(GRID_DIMS)), nrow=length(GRID_DIMS),ncol=length(NUM_REALIZATIONS))
rownames(outGPU) = GRID_DIMS
colnames(outGPU) = NUM_REALIZATIONS
outCPU = matrix(rep(0,length(NUM_REALIZATIONS)*length(GRID_DIMS)), nrow=length(GRID_DIMS),ncol=length(NUM_REALIZATIONS))
rownames(outCPU) = GRID_DIMS
colnames(outCPU) = NUM_REALIZATIONS

for (i in 1:length(GRID_DIMS)) {
	for (j in 1:length(NUM_REALIZATIONS)) {
		k = NUM_REALIZATIONS[j]
		n = GRID_DIMS[i]
		
		nx = n
		ny = n
		dx = (xmax-xmin)/nx
		dy = (ymax-ymin)/ny
		
		cat("GRID SIZE:")
		cat(n)
		cat("x")
		cat(n)
		cat("    k = ")
		cat(k)
		cat("\n")



		##### GPU PERFORMANCE #######
		timeGPU = 0	
		for (z in 1:ITERATIONS) {
			try(timeGPU <- timeGPU + system.time(simGPU <- gpuSim(GridTopology(c(xmin,ymin), c(dx,dy), c(nx,ny)),"Exp", sill,range,nugget, k))[3])
		}
		timeGPU = timeGPU / ITERATIONS
		cat("Average computation time GPU: ")
		cat(timeGPU)
		cat("s\n")
		outGPU[i,j] = timeGPU


		##### CPU PERFORMANCE fields #######
		timeCPU = 0	
		for (z in 1:ITERATIONS) {
			temp = proc.time()[3]
			grid<- list( x= seq( xmin,xmax,,nx), y= seq(ymin,ymax,,ny)) 
			obj<-Exp.image.cov( grid=grid, theta=1/range, setup=TRUE)
			for (t in 1:k){
				try(sim.rf(obj))
			}
			timeCPU = timeCPU + (proc.time()[3] - temp)
		}
		timeCPU = timeCPU / ITERATIONS
		cat("Average computation time CPU: ")
		cat(timeCPU)
		cat("s\n")
		outCPU[i,j] = timeCPU


		gc()
	}
}

return(list(gpu = outGPU, cpu = outCPU))
}




