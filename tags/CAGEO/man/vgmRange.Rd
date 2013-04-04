\name{vgmRange}
\alias{vgmRange}
\title{Variogram range parameter extraction}

\description{
	Extracts the range parameter of a variogramModel object for use in gpuSim(). If the variogramModel
	contains more than one model, the first no nugget model is chosen.  	
}

\usage{
	vgmRange(v) 
}

\arguments{
	\item{v}{Object of type variogramModel from gstat}
}

\value{Range parameter of the first no nugget model in v}

\examples{
	nugget = 4.5
	sill = 10
	range = 300
	x = vgm(sill, "Exp", range, nugget)
	
	vgmRange(x) # returns 300
}


\keyword{interpolation}
\keyword{variogram}
\keyword{kriging}
\keyword{covariance}
