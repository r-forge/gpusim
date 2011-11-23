\name{gpuBenchmark}
\alias{gpuBenchmark}
\title{Benchmarking the GPU for simulation of random fields compared to the fields package}

\description{
Performs an unconditional simulation using an exponential covariance function. The computation times for both, our gpusim package as well as the fields package are printed.
}

\usage{
gpuBenchmark(nx = 100, ny = 100, k = 100, range=2,sill=5,nugget=0)
}

\arguments{
	\item{nx}{number of grid columns}
	\item{ny}{number of grid rows}
	\item{k}{number of realizations}
	\item{range}{range parameter of variogram}
	\item{sill}{sill parameter of variogram}
	\item{nugget}{nugget parameter of variogram}
}

\value{Needed computation times are printed so that users can decide whether to use the gpu or cpu}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}


\examples{
library(gpusim)
gpuBenchmark()
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
