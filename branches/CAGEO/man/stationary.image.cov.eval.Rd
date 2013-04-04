\name{stationary.image.cov.eval}
\alias{stationary.image.cov.eval}
\title{
  See \code{?stationary.image.cov} in package \code{fields} but uses a user defined fft function. 
}
\description{
See \code{?stationary.image.cov} in package \code{fields}.
}
\usage{
stationary.image.cov.eval(ind1, ind2, Y, cov.obj, setup, grid, M, N, Covariance, Distance, fft_func, ...) 
}
\arguments{
\item{ind1}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{ind2}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{Y}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{cov.obj}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{setup}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{grid}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{M}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{N}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{Covariance}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{Distance}{
See \code{?stationary.image.cov} in package \code{fields}.
}
\item{fft_func}{The fft function to be used where the first argument represents the data values and the second attribute is a boolean value, which defines the direction (TRUE for inverse fft).}
\item{\dots}{
See \code{?stationary.image.cov} in package \code{fields}.
}

}
\value{
See \code{?stationary.image.cov} in package \code{fields}. This function prints the total computation time and the time needed for all ffts.
}

\examples{

library(fields)
library(gstat)

# meuse data
data(meuse)
coordinates(meuse) = ~x+y

out = Krig(coordinates(meuse), log(meuse$zinc), Covariance="Exponential", theta=500, rho=0.6, sigma2=0.05, m=1)

# simulation grid
xmin=178000
xmax=182000
ymin=329000
ymax=334000

x=seq(xmin+10, xmax, by=5)
y=seq(ymin+10, ymax, by=5)
length(x)
length(y)

grid=expand.grid(x,y)

sim.out = sim.Krig.grid.eval(out, grid.list =list(x,y), M=1, extrap=TRUE, fft_func=fft)
image(matrix(sim.out$z, nrow=length(sim.out$y), ncol=length(sim.out$x), byrow=TRUE))
}


