
\name{Exp.image.cov.eval}
\alias{Exp.image.cov.eval}
\title{
Exponential covariance function according to fields package but with a user defined fft function. 
}
\description{
See \code{?Exp.image.cov} in package \code{fields}.
}
\usage{
Exp.image.cov.eval(ind1, ind2, Y, cov.obj = NULL, setup = FALSE, grid, fft_func, ...)
}
\arguments{
\item{ind1}{
See \code{?Exp.image.cov} in package \code{fields}.
}
\item{ind2}{
See \code{?Exp.image.cov} in package \code{fields}.
}
\item{Y}{
See \code{?Exp.image.cov} in package \code{fields}.
}
\item{cov.obj}{
See \code{?Exp.image.cov} in package \code{fields}.
}
\item{setup}{
See \code{?Exp.image.cov} in package \code{fields}.
}
\item{grid}{
See \code{?Exp.image.cov} in package \code{fields}.
}
\item{fft_func}{The fft function to be used where the first argument represents the data values and the second attribute is a boolean value, which defines the direction (TRUE for inverse fft).}
\item{\dots}{
See \code{?Exp.image.cov} in package \code{fields}.
}

}

\value{
See \code{?Exp.image.cov} in package \code{fields}. This function prints the total computation time and the time needed for all ffts.
}
\details{
See \code{?Exp.image.cov} in package \code{fields}.
  
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

