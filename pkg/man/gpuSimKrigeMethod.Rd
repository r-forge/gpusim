\name{.gpuSimKrigeMethod}
\alias{.gpuSimKrigeMethod}
\title{Internal handling of kriging methods for conditioning}

\description{
Returns an internal integer representation of the given kriging method.
}

\usage{
	.gpuSimKrigeMethod(identifier) 
}

\arguments{
	\item{identifier}{An string itentifier for the kriging method: 'o' for ordinary, 's' for simple kriging.}
}


\value{An integer method code specified in C code}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
.gpuSimKrigeMethod('s') # <- 0
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}
