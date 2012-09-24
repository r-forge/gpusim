


gpuSSD <- function(a, b) {
	if (any(dim(a) != dim(b))) stop("Dimensions of input datasets do not match!")
	n = prod(dim(a))
	res = .C("gpuDist", out = double(1), as.double(a), as.double(b),  as.integer(n), DUP=FALSE, PACKAGE="gpusim")
	return(res$out)
}


cpuSSDMatrix <- function(a) {
	if (length(dim(a)) != 3) stop("Dimensions of input dataset is not equal to 3!")
	res = .C("cpuDistMatrix", out = double(dim(a)[3]*dim(a)[3]), as.double(a),  as.integer(dim(a)[1]), as.integer(dim(a)[2]), as.integer(dim(a)[3]), DUP=FALSE, PACKAGE="gpusim")
	return(matrix(res$out,nrow=dim(a)[3],ncol=dim(a)[3]))
}