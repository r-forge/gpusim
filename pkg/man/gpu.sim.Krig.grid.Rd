% fields, Tools for spatial data
% Copyright 2004-2011, Institute for Mathematics Applied Geosciences
% University Corporation for Atmospheric Research
% Licensed under the GPL -- www.gpl.org/licenses/gpl.html

\name{gpu.sim.Krig.grid}
\alias{gpu.sim.Krig.grid}
\title{
  See \code{?sim.Krig.grid} in package \code{fields}.
}
\description{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\usage{
gpu.sim.rf(object, grid.list, M, nx, ny, xy, verbose, sigma2, rho, extrap) 
}
\arguments{
\item{object}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{grid.list}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{M}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{nx}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{ny}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{xy}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{verbose}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{sigma2}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{rho}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\item{extrap}{
See \code{?sim.Krig.grid} in package \code{fields}.
}
}
\value{
See \code{?sim.Krig.grid} in package \code{fields}.
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

