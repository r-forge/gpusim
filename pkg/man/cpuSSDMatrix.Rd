\name{cpuSSDMatrix}
\alias{cpuSSDMatrix}
\title{Calculation of pairwise SSDs for three-dimensional arrays}

\description{
Computes the pairwise sum of squared differences of an image stack in a three-dimensional array where the last dimension denotes the number of images.
}

\usage{
	cpuSSDMatrix(a)
}

\arguments{
	\item{a}{a three-dimensional array (e.g. image stack)}
}

\value{A matrix containing the pairwise SSD values between single images}

\author{Marius Appel \email{marius.appel@uni-muenster.de}}

\examples{
# load a NetCDF file and compute ssd matrix between all image pairs
library(ncdf)
library(gpusim)

nc = open.ncdf("C:\\Users\\Marius\\Downloads\\detrended_zo_anomalies_1992_2008.nc")
data <- get.var.ncdf(nc)
close.ncdf(nc)

cpuSSDMatrix(data)
}

