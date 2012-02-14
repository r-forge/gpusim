\name{dCov2d}
\alias{dCov2d}
\title{Computation of a covariance matrix from given 2d points}

\description{
Computes the covariance matrix of two-dimensional points based on their pairwise distances and a given covariance function.
}

\usage{
	dCov2d(data, model, sill, range, nugget) 
}

\arguments{
	\item{data}{a matrix with 2 columns representing x and y coordinates, each row corresponds to a point}
	\item{model}{a string identifier for the covariance function to be used, currently \emph{Exp}, \emph{Gau} or  \emph{Sph}. See ?gpusim for more details.}
	\item{range}{range parameter of the covariance function}
	\item{sill}{sill parameter of the covariance function}
	\item{nugget}{nugget parameter of the covariance function}
}


\value{Covariance values for each pair of points in a symetric positive-definite matrix}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
library(gpusim)
x = 1:5
y = 1:5
xy = as.matrix(expand.grid(x,y))
dCov2d(xy,"Exp",1,2,0)
}

\keyword{random fields}
\keyword{covariance}

