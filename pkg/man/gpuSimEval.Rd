\name{gpuSimEval}
\alias{gpuSimEval}
\title{Evaluation of computation times needed for gpuSim()}

\description{
Performs exactly the same computations as gpuSim() but additionally returns needed computation times for different simulation steps.
}

\usage{
gpuSimEval <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE,prec.double=FALSE)
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
	\item{verify}{If TRUE, an average experimental variogram of the simulation results is generated and plotted together with the theoretical variogram model. Depending on the simulation dimensions, this may take some time. Using verification forces as.sp = TRUE.}
	\item{prec.double}{If TRUE, double precision is used. Notice that the performance for double precision strongly depends on the available hardware.}
}

\details{
This is a wrapper function for different types of simulation routines such as unconditional and conditional simulation. With regard to the given arguments, three different cumputations can be performed:
\enumerate{
         \item If \emph{uncond} is given, only the conditioning of given realizations will be performed. Note that \emph{samples} must be given too and \emph{k} is derived from \emph{samples}, it will be ignored if given.
         \item If \emph{uncond} is not given, but \emph{k} and \emph{samples} are given, a full conditional simulation including computing unconditional realizations will be performed.
		 \item If \emph{uncond} as well as \emph{samples} is missing but \emph{k} is given, k unconditional realizations will be computed.
       }
}

\value{
The result is a list containing exactly two elements:
	\enumerate{
         \item \emph{result} gives the result or the simulation as in \emph{gpusim}.
		 \item \emph{runtimes} are the needed computation times for different steps of the simulation given as a named numeric vector.
	}
	Note that all times are given in seconds.
}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}


\examples{
library(gpusim)

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
out = gpuSimEval(grid, model, sill, range, nugget, k)
image(out$result[,,1]) # plot first realization
print(out$runtimes)    # print computation times

# unconditional simulation with sp interoperability, probably slower
out = gpuSimEval(grid, model, sill, range, nugget, k, as.sp = TRUE)
image(out$result["sim1"]) # plot first realization
print(out$runtimes)    # print computation times
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}

