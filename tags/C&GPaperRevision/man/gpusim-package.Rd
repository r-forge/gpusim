\name{gpusim-package}
\alias{gpusim-package}
\alias{gpusim}
\docType{package}
\title{
Simulation of Gaussian Random Fields using NVIDIAs CUDA}

\description{
Performs (un)conditional simulation of gaussian random fields using NVIDIAs CUDA for GPU execution. 
}
\details{
\tabular{ll}{
Package: \tab gpusim\cr
Type: \tab Package\cr
Version: \tab 0.0.1\cr
Date: \tab 2012-06-19\cr
License: \tab GPL-3\cr
LazyLoad: \tab yes\cr
}
You can find a short tutorial about getting started with the package as a pdf document in the package's inst directory or on our website. 
}
\author{
Katharina Henneboehl, Marius Appel
Maintainer: Katharina Hennboehl <katharina.henneboehl@uni-muenster.de>
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}


\examples{
library(gpusim)
library(fields) # for image.plot

# 2d simulation


# build grid
xmin = 0
xmax = 5
ymin = 0
ymax = 5
nx = 100
ny = 100
dx = (xmax-xmin)/nx
dy = (ymax-ymin)/ny
grid = GridTopology(c(xmin,ymin), c(dx,dy), c(nx,ny))

# covariance arguments
model = "Sph"
range = 0.5
sill = 3
nugget = 0
anis = c(45,0.6) # anisotropic model

k = 5  ## number of realizations


# unconditional simulation
simGPU = gpuSim(grid, model, sill, range, nugget, k, anis=anis)
image.plot(simGPU[,,5]) ## plot 5-th realization

# unconditional simulation with sp interoperability
simGPUsp = gpuSim(grid, model, sill, range, nugget, k, as.sp = TRUE, anis=anis)
spplot(simGPUsp) # plot all realizations using sp



# build random samples for conditional simulation
n = 100 ## number of samples
sample_x = runif(n,min=xmin,max=xmax) # sample x coords
sample_y = runif(n,min=ymin,max=ymax) # sample y coords
sample_z = rnorm(n,20,4) ## sample values
samples = data.frame(x=sample_x, y=sample_y)
coordinates(samples)=c("x","y")
samples = SpatialPointsDataFrame(samples,as.data.frame(sample_z))

# conditional simulation without sp interoperability using ordinary kriging
simGPUcond = gpuSim(grid, model, sill, range, nugget, k, samples, anis=anis)
image.plot(simGPUcond[,,1]) ## plot 5-th realization

# conditional simulation with sp interoperability using ordinary kriging
simGPUcondsp = gpuSim(grid, model, sill, range, nugget, k, samples, as.sp = TRUE)
spplot(simGPUcondsp) # plot all realizations using sp


# conditional simulation using simple kriging assuming a WRONG mean value 5 (20 would be correct)
simGPUcond = gpuSim(grid, model, sill, range, nugget, k, samples, kriging.method='s',mu=5, anis=anis)
image.plot(simGPUcond[,,5]) ## plot 5-th realization


# conditioning by ordinary kriging using unconditional realizations computed before
simGPUcond1 = gpuSim(grid,model,sill,range,nugget,samples=samples,uncond=simGPU, anis=anis)
image.plot(simGPUcond1[,,5])





# 3d simulation

# unconditional isotropic

# build grid
xmin = 0
xmax = 10
ymin = 0
ymax = 10
zmin = 0
zmax = 10
nx = 100
ny = 100
nz = 100
dx = (xmax-xmin)/nx
dy = (ymax-ymin)/ny
dz = (zmax-zmin)/nz
grid = GridTopology(c(xmin,ymin,zmin), c(dx,dy,dz), c(nx,ny,nz))

model = "Sph"
range = 2
sill = 0.1
nugget = 0.0

k = 2  ## number of realizations

simGPU = gpuSim(grid, model, sill, range, nugget, k, prec.double=T)

# animation of second realization
for (i in 1:dim(simGPU)[3]) {
	image.plot(simGPU[,,i,2], zlim=c(-1.5,1.5))
	Sys.sleep(0.1)
}



# conditional simulation with anisotropic covariance model

anis=c(45,45,45,0.9,0.3)
k = 2

# build random samples for conditional simulation
n = 20 ## number of samples
sample_x = runif(n,min=xmin,max=xmax) # sample x coords
sample_y = runif(n,min=ymin,max=ymax) # sample y coords
sample_z = runif(n,min=zmin,max=zmax) # sample y coords
sample_data = rnorm(n,20,4) ## sample values

# use outlier data point at the cube's center
sample_data[n] = 100
sample_x[n] = 5
sample_y[n] = 5
sample_z[n] = 5

samples = data.frame(x=sample_x, y=sample_y, z=sample_z)
coordinates(samples)=c("x","y","z")
samples = SpatialPointsDataFrame(samples,as.data.frame(sample_data))

simGPUcond = gpuSim(grid, model, sill, range, nugget, k, samples,check=F,anis=anis,prec.double=T)

# animation of second realization
for (i in 1:dim(simGPUcond)[3]) {
	image.plot(simGPUcond[,,i,2], zlim=c(10,100), main=paste("z = ",xmin+(i-1)*dy))
	Sys.sleep(0.1)
}

}
