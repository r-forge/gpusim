\name{vgmAnis}
\alias{vgmAnis}
\title{Variogram anisotropy parameter extraction}

\description{
	Extracts the anisotropy parameter of a variogramModel object for use in gpuSim(). 
}

\usage{
	vgmAnis(v) 
}

\arguments{
	\item{v}{Object of type variogramModel from gstat}
}

\value{Anisotropy parameter of the model}

\examples{
	nugget = 4.5
	sill = 10
	range = 300
	anis = c(30,0.3)
	x = vgm(sill, "Exp", range, nugget)
	
	vgmAnis(x) # returns c(30,0,0,0.3,1)
}


\keyword{interpolation}
\keyword{variogram}
\keyword{kriging}
\keyword{covariance}
