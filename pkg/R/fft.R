
gpuFFTplan <- function(nx, ny=1, nz=1) {
	res = .C("planFFT", as.integer(nx), as.integer(ny), as.integer(nz), DUP=FALSE, PACKAGE="gpusim")
}



gpuFFT <- function(x, inv=FALSE) {
	d = 0
	if (class(x) == "array" || class(x) == "matrix") {
		d = dim(x)
	}
	else {
		d = c(length(x))
	}
	n = prod(d)
	nx = d[1]
	ny = 1
	nz = 1
	if (length(d) > 1) ny = d[2]
	if (length(d) > 2) nz = d[3]
	iscomplex = is.complex(x)
	
	res = .C("execFFT", out = complex(n),x , as.integer(nx), as.integer(ny), as.integer(nz), as.integer(inv),as.integer(iscomplex), DUP=FALSE, PACKAGE="gpusim")
	dim(res$out) = d
	return(res$out)
}
