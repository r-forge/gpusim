\name{.covID}
\alias{.covID}
\title{Internal ID representation for covariance functions}

\description{
Returns the internal ID of a given covariance function.
}

\usage{
	.covID(covmodel) 
}

\arguments{
	\item{covmodel}{A character sequence identifying the covariance function. Currently, \emph{Exp}, \emph{Sph} or \emph{Gau}}
}


\value{Internal ID for given covariance model}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
.covID("Sph") 
}

\keyword{cuda}
\keyword{gpu}
\keyword{simulation}
\keyword{random fields}
\keyword{covariance}
