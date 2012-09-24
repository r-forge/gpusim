\name{gpuSSD}
\alias{gpuSSD}
\title{Sum of squared differences on GPUs}

\description{
Computes the sum of squared differences of an array using the GPU.
}

\usage{
	gpuSSD(a,b) 
}

\arguments{
	\item{a}{the first array}
	\item{b}{the second array}
}

\value{The sum of squared differences of a and b.}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
# load a NetCDF file and compute a ssd value of two images
library(ncdf)
library(gpusim)

nc = open.ncdf("C:\\Users\\Marius\\Downloads\\detrended_zo_anomalies_1992_2008.nc")
data <- get.var.ncdf(nc)
close.ncdf(nc)

# reduce size for testing
data = data[,,1:10]

# similarity between i-th and j-th image
i = 4
j = 5
sum((data[,,i] - data[,,j])^2) # for comparison
gpuSSD(data[,,i],data[,,j])
}


