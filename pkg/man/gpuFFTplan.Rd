\name{gpuFFTplan}
\alias{gpuFFTplan}
\title{Planning a Fast-Fourier-Transformation using NVIDIA's cufft library}

\description{
Creates a cufft plan and stores it in an internal plan buffer.
}

\usage{
	gpuFFTplan(nx,ny=1,nz=1) 
}

\arguments{
	\item{nx}{size in first dimension}
	\item{ny}{size in second dimension}
	\item{nz}{size in third dimension}
	\item{prec.double}{If TRUE, double precision is used instead of single precision}
}


\value{None - cufft plan is stored internally}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
library(gpusim)
dims = c(100,100,1000)
# generate random data with specified dimensions
data = array(runif(prod(dims)),dim=dims)
# plan fft
gpuFFTplan(dims[1],dims[2],dims[3]) # optionally, if not the plan is created by gpuFFT()
# compare gpuFFT() and fft()
system.time(x <- gpuFFT(data))
system.time(y <- fft(data))
}

\keyword{cuda}
\keyword{gpu}
\keyword{cufft}
\keyword{fft}