


gpudist <- function(a, b) {
	if (any(dim(a) != dim(b))) stop("Dimensions of input datasets do not match!")
	n = prod(dim(a))
	res = .C("gpuDist", out = double(1), as.double(a), as.double(b),  as.integer(n), DUP=FALSE, PACKAGE="gpusim")
	return(res$out)
}