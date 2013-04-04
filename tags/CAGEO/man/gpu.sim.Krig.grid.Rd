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

sim.out = gpu.sim.Krig.grid(out, grid.list =list(x,y), M=1, extrap=TRUE)
image(matrix(sim.out$z, nrow=length(sim.out$y), ncol=length(sim.out$x), byrow=TRUE))
}

