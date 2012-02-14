\name{gpuSimCatchError}
\alias{gpuSimCatchError}
\title{Internal error handling of gpu simulation}

\description{
Returns an error message for an internal specified integer code.
}

\usage{
	gpuSimCatchError(code) 
}

\arguments{
	\item{code}{An integer error code specified in C code}
}


\value{An error message}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
gpuSimCatchError(0) # <- OK
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}
