\name{covSpherical}
\alias{covSpherical}
\title{Sphericall covariance function}

\description{
Evaluates a spherical covariance function to a given distance array.
}

\usage{
	covSpherical(data, sill, range, nugget) 
}

\arguments{
	\item{data}{distance values}
	\item{range}{range parameter of variogram}
	\item{sill}{sill parameter of variogram}
	\item{nugget}{nugget parameter of variogram}
}

\details{
The spherical covariance function is defined as \eqn{C(h) = sill \cdot (1 - (\frac{3h}{2 \cdot range} - \frac{h^3}{2 \cdot range^3}))}{C(h) = sill * (1 - ((3h/2*range) - (h^3 / 2*range^3))} for 0 < h <= range, \eqn{C(0) = nugget + sill} and \eqn{C(h) = 0} for h > range.
}

\value{Covariance values for each item of data}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
dist = c(1,2,3,4,5)
covs = covSpherical(dist, 2.0, 3.0, 1.0) 
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}
