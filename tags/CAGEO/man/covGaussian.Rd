\name{covGaussian}
\alias{covGaussian}
\title{Gaussian covariance function}

\description{
Evaluates a gaussian covariance function to a given distance array.
}

\usage{
	covGaussian(data, sill, range, nugget) 
}

\arguments{
	\item{data}{distance values}
	\item{range}{range parameter of variogram}
	\item{sill}{sill parameter of variogram}
	\item{nugget}{nugget parameter of variogram}
}

\details{
The gaussian covariance function is defined as \eqn{C(h) = sill \cdot exp(-(\frac{h}{range})^{2})}{C(h) = sill * exp(-(h/range)^2)} where \eqn{C(0) = nugget + sill}.
}

\value{Covariance values for each item of data}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
dist = c(1,2,3,4,5)
covs = covGaussian(dist, 2.0, 3.0, 1.0) 
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}

