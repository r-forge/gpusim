\name{covExponential}
\alias{covExponential}
\title{Exponential covariance function}

\description{
Evaluates a exponential covariance function to a given distance array.
}

\usage{
	covExponential(data, sill, range, nugget) 
}

\arguments{
	\item{data}{distance values}
	\item{range}{range parameter of variogram}
	\item{sill}{sill parameter of variogram}
	\item{nugget}{nugget parameter of variogram}
}

\details{
The exponential covariance function is defined as \eqn{C(h) = sill * exp(-h/range)} where \eqn{C(0) = nugget + sill}.
}

\value{Covariance values for each item of data}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
dist = c(1,2,3,4,5)
covs = covExponential(dist, 2.0, 3.0, 1.0) 
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}
