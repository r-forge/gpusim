\name{vgmModel}
\alias{vgmModel}
\title{Variogram sill parameter extraction}

\description{
	Extracts the sill parameter of a variogramModel object for use in gpuSim(). If the variogramModel
	contains more than one model, the first no nugget model is chosen.  	
}

\usage{
	vgmSill(v) 
}

\arguments{
	\item{v}{Object of type variogramModel from gstat}
}

\value{Sill value of the first no nugget model in v}

\examples{
	nugget = 4.5
	sill = 10
	range = 300
	x = vgm(sill, "Exp", range, nugget)
	
	vgmSill(x) # returns 10
}


\keyword{interpolation}
\keyword{variogram}
\keyword{kriging}
\keyword{covariance}
