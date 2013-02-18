\name{vgmNugget}
\alias{vgmNugget}
\title{Variogram nugget effect extraction}

\description{
	Extracts the nugget effect value of a variogramModel object for use in gpuSim(). If the variogramModel
	contains more than one nugget models, the effects are summed up.  	
}

\usage{
	vgmNugget(v) 
}

\arguments{
	\item{v}{Object of type variogramModel from gstat}
}

\value{Sum of all nugget models in v}

\examples{
	nugget = 4.5
	sill = 10
	range = 300
	x = vgm(sill, "Exp", range, nugget)
	
	vgmNugget(x) # returns 4.5
}


\keyword{interpolation}
\keyword{variogram}
\keyword{kriging}
\keyword{covariance}
