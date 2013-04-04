\name{sim.Krig.grid.eval}
\alias{sim.Krig.grid.eval}
\title{
  See \code{?sim.Krig.grid} in package \code{fields} but uses a user defined fft function
}
\description{
See \code{?sim.Krig.grid} in package \code{fields}.
}
\usage{
sim.Krig.grid.eval(object, grid.list, M, nx, ny, xy, verbose, sigma2, rho, extrap, fft_func) 
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
\item{fft_func}{The fft function to be used where the first argument represents the data values and the second attribute is a boolean value, which defines the direction (TRUE for inverse fft).}


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

sim.out = sim.Krig.grid.eval(out, grid.list =list(x,y), M=1, extrap=TRUE, fft_func=fft)
image(matrix(sim.out$z, nrow=length(sim.out$y), ncol=length(sim.out$x), byrow=TRUE))
}

