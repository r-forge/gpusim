
\name{gpu.Exp.image.cov}
\alias{gpu.Exp.image.cov}
\title{
Exponential covariance function according to fields package but with fft performed on graphics processing units. 
}
\description{
See \code{?Exp.image.cov} in package \code{fields}.
}
\usage{
gpu.Exp.image.cov(ind1, ind2, Y, cov.obj = NULL, setup = FALSE, grid, ...)
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
\item{\dots}{
See \code{?Exp.image.cov} in package \code{fields}.
}
}

\value{
See \code{?Exp.image.cov} in package \code{fields}.
}
\details{
See \code{?Exp.image.cov} in package \code{fields}.
  
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

