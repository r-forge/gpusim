\name{sim.rf.eval}
\alias{sim.rf.eval}
\title{
  Simulates a random field using the \code{fields} package but a user defined fft function. 
}
\description{
See \code{?sim.rf} in package \code{fields}.
}
\usage{
gpu.sim.rf(obj, fft_func)
}
\arguments{
\item{obj}{See \code{?sim.rf} in package \code{fields}.}
\item{fft_func}{The fft function to be used where the first argument represents the data values and the second attribute is a boolean value, which defines the direction (TRUE for inverse fft).}
}
\value{
See \code{?sim.rf} in package \code{fields}. This function prints the total computation time and the time needed for all ffts.
}

\examples{
####################################################
## Performance Comparison ##########################

library(fields)

n = 1000

## gpu fields single prec
grid<- list( x= seq( 0,5,,n), y= seq(0,5,,n)) 
obj <- Exp.image.cov.eval( grid=grid, theta=.5, setup=TRUE,fft_func = gpuFFT)
res <- sim.rf.eval(obj,fft_func = gpuFFT)
image.plot(res)

## gpu fields double prec
gpuFFTd <- function(x, inv=FALSE) return(gpuFFT(x,inv,T))
grid<- list( x= seq( 0,5,,n), y= seq(0,5,,n)) 
obj <- Exp.image.cov.eval( grid=grid, theta=.5, setup=TRUE,fft_func = gpuFFTd)
res <- sim.rf.eval(obj,fft_func = gpuFFTd)
image.plot(res)

## cpu fields
grid<- list( x= seq( 0,5,,n), y= seq(0,5,,n)) 
obj <- Exp.image.cov.eval( grid=grid, theta=.5, setup=TRUE,fft_func = fft)
res <- sim.rf.eval(obj,fft_func = fft)
image.plot(res)
}

