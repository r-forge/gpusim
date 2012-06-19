\name{gpuSim}
\alias{gpuSim}
\title{Simulation of Gaussian Random Fields using CUDA}

\description{
Performs (un)conditional simulation of gaussian random fields using NVIDIAs CUDA for GPU execution.
}

\usage{
gpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, benchmark = FALSE, prec.double=FALSE, anis=c(0,0,0,1,1))
}

\arguments{
	\item{grid}{Either a GridTopology or a SpatialPixelsDataFrame object containing the underlying grid}
	\item{covmodel}{A character sequence identifying the covariance function. Note that the functions do not include anisotropy information and are described by three arguments \emph{sill}, \emph{range} and \emph{nugget}. Currently, the following functions are implemented: 
	\itemize{
         \item \emph{Exp} - exponential covariance function: \eqn{C(h) = sill \cdot exp(\frac{-h}{range})}{C(h) = sill * exp(-h/range)} where \eqn{C(0) = nugget + sill}
         \item \emph{Gau} - gaussian covariance function: \eqn{C(h) = sill \cdot exp(-(\frac{h}{range})^{2})}{C(h) = sill * exp(-(h/range)^2)} where \eqn{C(0) = nugget + sill} 
		 \item \emph{Sph} - spherical covariance function:  \eqn{C(h) = sill \cdot (1 - (\frac{3h}{2 \cdot range} - \frac{h^3}{2 \cdot range^3}))}{C(h) = sill * (1 - ((3h/2*range) - (h^3 / 2*range^3))} for 0 < h <= range, \eqn{C(0) = nugget + sill} and \eqn{C(h) = 0} for h > range.
		 \item \emph{Mat3} - matern covariance function with v=3/2: \eqn{C(h) = sill \cdot (1 + \frac{\sqrt{3}h}{range}) \cdot exp(- \frac{\sqrt{3}h}{range})} where \eqn{C(0) = nugget + sill}.
         \item \emph{Mat5} - matern covariance function with v=5/2: \eqn{C(h) = sill \cdot (1 + \frac{\sqrt{5}h}{range} + \frac{5h^{2}}{3 \cdot range^{2}}) \cdot exp(- \frac{\sqrt{5}h}{range})} where \eqn{C(0) = nugget + sill}.     
	   }}
	\item{sill}{sill parameter of covariance function}	
	\item{range}{range parameter of covariance function}
	\item{nugget}{nugget parameter of covariance function}
	\item{k}{Number of realizations}
	\item{samples}{Sample points as a SpatialPointsDataFrame object containing exactly one data field which is used for conditioning. If this argument is missing, only unconditional simulation will be performed.}
	\item{uncond}{Realizations of an unconditional simulation. Used together with \emph{samples} to condition previously computed realizations of an unconditional simulation using real sample data. The realizations can be given either as a three-dimensional array with the realization as third index or as a SpatialPixelsDataFrame object containing realizations as data fields. Note that you have to give the same covariance function which was used for generating the unconditional simulation in order to get meaningful results.}	
	\item{kriging.method}{String identifier of the kriging method used for conditioning. Currently 's' or 'S' for simple kriging and 'o' or 'O' for ordinary kriging. Notice that for simple kriging, a known mean value can be given in mu.}
	\item{mu}{Mean value for conditioning the field using simple kriging. If ordinary kriging is used, this argument is unused.}
	\item{as.sp}{TRUE or FALSE, should result be returned as SpatialPixelsDataFrame or as an simple array?}
	\item{check}{TRUE or FALSE, should fft of covariance matrix be checked for negative real parts?}
	\item{benchmark}{If TRUE, detailed information about computation times of internally called functions is given.}
	\item{prec.double}{If TRUE, double precision is used. Notice that the performance for double precision strongly depends on the available hardware.}
	\item{anis}{Anisotropy argument, see ?vgm from gstat package for further information}

}

\details{
This is a wrapper function for different types of simulation routines such as unconditional and conditional simulation. With regard to the given arguments, three different cumputations can be performed:
\enumerate{
         \item If \emph{uncond} is given, only the conditioning of given realizations will be performed. Note that \emph{samples} must be given too and \emph{k} is derived from \emph{samples}, it will be ignored if given.
         \item If \emph{uncond} is not given, but \emph{k} and \emph{samples} are given, a full conditional simulation including computing unconditional realizations will be performed.
		 \item If \emph{uncond} as well as \emph{samples} is missing but \emph{k} is given, k unconditional realizations will be computed.
       }
}

\value{As written above, the result is either a set of unconditional or conditional realizations of the specified grid and covariance function.	
Note that the \emph{as.sp} argument determines the class of the result. If it is set to TRUE, a SpatialPixelsDataFrame object containing the realizations as different columns in the data attribute will be returned. Otherwise, a raw array is returned where the last index denotes the realization.
If returned as SpatialPixelsDataFrame object, each realization is one column of the object's \emph{data} slot.
}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}


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

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
