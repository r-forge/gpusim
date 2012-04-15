% fields, Tools for spatial data
% Copyright 2004-2011, Institute for Mathematics Applied Geosciences
% University Corporation for Atmospheric Research
% Licensed under the GPL -- www.gpl.org/licenses/gpl.html

\name{sim.rf}
\alias{sim.rf}
\title{
  Simulates a random field using the \code{fields} package but a GPU accelerated fft. 
}
\description{
See \code{?sim.rf} in package \code{fields}.
}
\usage{
sim.rf(obj)
}
\arguments{
\item{obj}{
See \code{?sim.rf} in package \code{fields}.
}
\item{\dots}{
See \code{?sim.rf} in package \code{fields}.}

}
\value{
See \code{?sim.rf} in package \code{fields}.
}

\examples{
####################################################
## Performance Comparison ##########################

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

