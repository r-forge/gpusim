\name{gpuFFT}
\alias{gpuFFT}
\title{Executing a Fast-Fourier-Transformation using NVIDIA's cufft library}

\description{
Runs a FFT using NVIDIA's cufft library. Looks, if a plan has already been created. If not, it is created.
}

\usage{
	gpuFFT(x, inv = FALSE,, prec.double = FALSE) 
}

\arguments{
	\item{x}{up to 3 dimensional numeric data}
	\item{inv}{TRUE or FALSE, should the inverse FFT be executed}
	\item{prec.double}{If TRUE, double precision is used instead of single precision}
}

\value{Complex array of spectral coefficients}

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