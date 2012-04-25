% fields, Tools for spatial data
% Copyright 2004-2011, Institute for Mathematics Applied Geosciences
% University Corporation for Atmospheric Research
% Licensed under the GPL -- www.gpl.org/licenses/gpl.html

\name{gpu.stationary.image.cov}
\alias{gpu.stationary.image.cov}
\title{
  See \code{?stationary.image.cov} in package \code{fields}. 
}
\description{
See \code{?stationary.image.cov} in package \code{fields}.
}
\usage{
gpu.stationary.image.cov(ind1, ind2, Y, cov.obj, setup, grid, M, N, Covariance, Distance, ...) 
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
\item{\dots}{
See \code{?stationary.image.cov} in package \code{fields}.
}

}
\value{
See \code{?stationary.image.cov} in package \code{fields}.
}

\examples{
####################################################
## Performance Comparison ##########################

library(fields)

n = 1000

# fields
grid<- list( x= seq( 0,5,,n), y= seq(0,5,,n)) 
system.time(obj <- Exp.image.cov( grid=grid, theta=.5, setup=TRUE))
system.time(res <- sim.rf(obj))
image.plot(res)


# gpu fields
grid<- list( x= seq( 0,5,,n), y= seq(0,5,,n)) 
system.time(obj <- gpu.Exp.image.cov( grid=grid, theta=.5, setup=TRUE))
system.time(res <- gpu.sim.rf(obj))
image.plot(res)


# gpusim
grid = GridTopology(c(0,0), c(5/n,5/n), c(n,n))
system.time(simGPU <- gpuSim(grid, "Exp", 1, 0.5, 0, 1))
image.plot(simGPU[,,1])
}

