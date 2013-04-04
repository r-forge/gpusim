\name{covMatern5}
\alias{covMatern5}
\title{Matern covariance function with v=5/2}

\description{
Applies the matern covariance with v=5/2 function to a given distance array.
}

\usage{
	covMatern5(data, sill, range, nugget) 
}

\arguments{
	\item{data}{distance values}
	\item{range}{range parameter of variogram}
	\item{sill}{sill parameter of variogram}
	\item{nugget}{nugget parameter of variogram}
}

\details{
The matern covariance function with v=5/2 is defined as \eqn{C(h) = sill \cdot (1 + \frac{\sqrt{5}h}{range} + \frac{5h^{2}}{3 \cdot range^{2}}) \cdot exp(- \frac{\sqrt{5}h}{range})} where \eqn{C(0) = nugget + sill}.
}

\value{Covariance values for each item of data}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
dist = c(1,2,3,4,5)
covs = covMatern5(dist, 2.0, 3.0, 1.0) 
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}
