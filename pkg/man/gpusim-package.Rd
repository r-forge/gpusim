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
Date: \tab 2012-02-25\cr
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

gpuDeviceInfo()

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
model = "Gau"
range = 0.5
sill = 3
nugget = 0

k = 5  ## number of realizations


# unconditional simulation
simGPU = gpuSim(grid, model, sill, range, nugget, k)
image.plot(simGPU[,,5]) ## plot 5-th realization

# unconditional simulation with sp interoperability
simGPUsp = gpuSim(grid, model, sill, range, nugget, k, as.sp = TRUE)
spplot(simGPUsp) # plot all realizations using sp



# build random samples for conditional simulation
n = 100 ## number of samples
sample_x = runif(n,min=xmin,max=xmax) # sample x coords
sample_y = runif(n,min=ymin,max=ymax) # sample y coords
sample_z = rnorm(n,20,4) ## sample values
samples = data.frame(x=sample_x, y=sample_y)
coordinates(samples)=c("x","y")
samples = SpatialPointsDataFrame(samples,as.data.frame(sample_z))

# conditional simulation without sp interoperability
simGPUcond = gpuSim(grid, model, sill, range, nugget, k, samples)
image.plot(simGPUcond[,,1]) ## plot 5-th realization

# conditional simulation with sp interoperability
simGPUcondsp = gpuSim(grid, model, sill, range, nugget, k, samples, as.sp = TRUE)
spplot(simGPUcondsp) # plot all realizations using sp



# conditioning using unconditional realizations computed before
simGPUcond1 = gpuSim(grid,model,sill,range,nugget,samples=samples,uncond=simGPU)
image.plot(simGPUcond1[,,5])

}
