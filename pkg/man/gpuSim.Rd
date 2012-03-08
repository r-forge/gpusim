\name{gpuSim}
\alias{gpuSim}
\title{Simulation of Gaussian Random Fields using CUDA}

\description{
Performs (un)conditional simulation of gaussian random fields using NVIDIAs CUDA for GPU execution.
}

\usage{
gpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE)
}

\arguments{
	\item{grid}{Either a GridTopology or a SpatialPixelsDataFrame object containing the underlying grid}
	\item{covmodel}{A character sequence identifying the covariance function. Note that the functions do not include anisotropy information and are described by three arguments \emph{sill}, \emph{range} and \emph{nugget}. Currently, the following functions are implemented: 
	\itemize{
         \item \emph{Exp} - exponential covariance function \eqn{C(h) = sill \cdot exp(\frac{-h}{range})}{C(h) = sill * exp(-h/range)} where \eqn{C(0) = nugget + sill}
         \item \emph{Gau} - gaussian covariance function \eqn{C(h) = sill \cdot exp(-(\frac{h}{range})^{2})}{C(h) = sill * exp(-(h/range)^2)} where \eqn{C(0) = nugget + sill} 
		 \item \emph{Sph} - spherical covariance function  \eqn{C(h) = sill \cdot (1 - (\frac{3h}{2 \cdot range} - \frac{h^3}{2 \cdot range^3}))}{C(h) = sill * (1 - ((3h/2*range) - (h^3 / 2*range^3))} for 0 < h <= range, \eqn{C(0) = nugget + sill} and \eqn{C(h) = 0} for h > range.
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
	\item{verify}{If TRUE, an average experimental variogram of the simulation results is generated and plotted together with the theoretical variogram model. Depending on the simulation dimensions, this may take some time. Using verification forces as.sp = TRUE.}
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
Note that the \emph{as.sp} argument determines the class of the result. If it is set to TRUE, a SpatialPixelsDataFrame object containing the realizations as different columns in the data attribute will be returned. Otherwise, a three-dimensional raw array is returned where the third index denotes the realization, the second denotes the row and the first denotes the column in the grid.
If returned as SpatialPixelsDataFrame object, each realization is one column of the object's \emph{data} slot.
}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\note{
gpuSim currently does NOT check the fft covariance grid to be positive definite! Thus, you should make sure to select an appropriate covariance model.
}

\examples{
library(gpusim)
library(fields) # for image.plot

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

# conditional simulation without sp interoperability using ordinary kriging
simGPUcond = gpuSim(grid, model, sill, range, nugget, k, samples)
image.plot(simGPUcond[,,1]) ## plot 5-th realization

# conditional simulation with sp interoperability using ordinary kriging
simGPUcondsp = gpuSim(grid, model, sill, range, nugget, k, samples, as.sp = TRUE)
spplot(simGPUcondsp) # plot all realizations using sp


# conditional simulation using simple kriging assuming a WRONG mean value 5 (20 would be correct)
simGPUcond = gpuSim(grid, model, sill, range, nugget, k, samples, kriging.method='s',mu=5)
image.plot(simGPUcond[,,5]) ## plot 5-th realization


# conditioning by ordinary kriging using unconditional realizations computed before
simGPUcond1 = gpuSim(grid,model,sill,range,nugget,samples=samples,uncond=simGPU)
image.plot(simGPUcond1[,,5])

}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
