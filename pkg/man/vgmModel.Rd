\name{vgmModel}
\alias{vgmModel}
\title{Variogram model parameter extraction}

\description{
	Extracts the model string identifier of a variogramModel object for use in gpuSim(). If the variogramModel
	contains more than one model, the first no nugget model is chosen.  	
}

\usage{
	vgmModel(v) 
}

\arguments{
	\item{v}{Object of type variogramModel from gstat}
}

\value{String identifier of the first no nugget model in v}

\examples{
	nugget = 4.5
	sill = 10
	range = 300
	x = vgm(sill, "Exp", range, nugget)
	
	vgmModel(x) # returns "Exp"
}


\keyword{interpolation}
\keyword{variogram}
\keyword{kriging}
\keyword{covariance}
