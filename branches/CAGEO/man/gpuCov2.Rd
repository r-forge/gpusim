\name{gpuCov2}
\alias{gpuCov2}
\title{Computation of a covariance matrix from given two-dimensional points}

\description{
Computes the covariance matrix of two-dimensional points based on their pairwise distances and a given covariance function on GPU.
}

\usage{
	gpuCov2(data, model, sill, range, nugget, anis=c(0,0,0,1,1), prec.double=FALSE) 
}

\arguments{
	\item{data}{a matrix with 2 columns representing x and y coordinates, each row corresponds to a point}
	\item{model}{a string identifier for the covariance function to be used, currently \emph{Exp}, \emph{Gau} or  \emph{Sph}. See ?gpusim for more details.}
	\item{range}{range parameter of the covariance function}
	\item{sill}{sill parameter of the covariance function}
	\item{nugget}{nugget parameter of the covariance function}
	\item{anis}{Anisotropy argument, see ?vgm from gstat package for further information}
	\item{prec.double}{Double or single precision to be used}
}


\value{Covariance values for each pair of points in a symetric positive-definite matrix}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
library(gpusim)
x = 1:5
y = 1:5
xy = as.matrix(expand.grid(x,y))
gpuCov2(xy,"Exp",1,2,0)
}

\keyword{random fields}
\keyword{covariance}

