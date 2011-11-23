\name{gpuSimEval}
\alias{gpuSimEval}
\title{Evaluation of computation times needed for gpuSim()}

\description{
Performs exactly the same computations as gpuSim() but does not return any realiaztion but a detailed list of needed computation times for different tasks.
}

\usage{
gpuSimEval(grid, covmodel, sill, range, nugget, k, samples, uncond, as.sp = FALSE)
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
	\item{as.sp}{TRUE or FALSE, should result be returned as SpatialPixelsDataFrame or as an simple array?}
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
The result is a vector with exactly four elements:
	\enumerate{
         \item \emph{time.total} gives the total time needed for the simulation.
		 \item \emph{time.gpupreprocessing} is the time that is needed for doing preprocessing steps on the GPU which are only needed oncer for all realizations.
		 \item \emph{time.gpurealizations} is the time that is needed for generating all realizations.
		 \item \emph{time.remaining} returns the remaining time for doing sime pre- or postprocessing steps in R. For instance, this includes the creation of a sp object if as.sp = TRUE and computing the inverse covariance matrix of sample points in conditioning. 
    }
	Note that all times are given in seconds.
}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\note{
gpuSim currently does NOT check the fft covariance grid to be positive definite! Thus, you should make sure to select an appropriate covariance model.
}

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
gpuSimEval(grid, model, sill, range, nugget, k)

# unconditional simulation with sp interoperability, probably slower
gpuSimEval(grid, model, sill, range, nugget, k, as.sp = TRUE)
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}

