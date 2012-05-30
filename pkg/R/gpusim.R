 
 
 gpuDeviceInfo <- function() {
	cat("\n******** GPU DEVICE INFO ********\n")
	result = .C("deviceInfo", out=format("",width=255),PACKAGE="gpusim")
	cat(result$out)	
	cat("*********************************\n\n")
 }



gpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE, prec.double=FALSE) {
	if (prec.double) {
		return(.sim2d(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify))
	}
	else return(.sim2f(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify))
}


gpuSimEval <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE, prec.double=FALSE) {
	if (prec.double) {
		return(.sim2dEval(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify))
	}
	else return(.sim2fEval(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify))
}



## IDs must be equal to enumeration in C code
.covID <- function(covmodel) {
	if (covmodel == "Exp") return(0)
	else if (covmodel == "Sph") return(1)
	else if (covmodel == "Gau") return(2)
	else if (covmodel == "Mat3") return(3)
	else if (covmodel == "Mat5") return(4)
	else stop("Unknown covariance function!")
}



covExponential <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covExp", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)	
}

covGaussian <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covGau", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)
}

covSpherical <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covSph", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)
}


covMatern3 <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covMat3", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)
}

covMatern5 <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covMat5", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)
}



dCov2d <- function(data, model, sill, range, nugget) {
	if (class(data) != "matrix") {
		stop("Expected a matrix as input!")
	}
	n = nrow(data)
	if (ncol(data) != 2) {
		stop("Expected a matrix with exactly 2 columns!")
	}
	res = 0
	if (model == "Exp") res = .C("dCovExp_2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	else if (model == "Gau") res = .C("dCovGau_2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")	
	else if (model == "Sph") res = .C("dCovSph_2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	else if (model == "Mat3") res = .C("dCovMat3_2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	else if (model == "Mat5") res = .C("dCovMat5_2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	else stop("Unknown covariance model!")	
	dim(res$out) = c(n,n)
	return(res$out)
}





## Gets the first model identifier string which is not nugget
vgmModel <- function(x) {
	return(as.character(x[x$model != "Nug",]$model[1]))
}

## Gets sum of all nugget effects in vgm model 
vgmNugget <- function(x) {
	return(sum(x[x$model == "Nug",]$psill))
}

## Gets the sill value of the first model identifier string which is not nugget
vgmSill <- function(x) {
	return(x[x$model != "Nug",]$psill[1])
}

## Gets the range value of the first model identifier string which is not nugget
vgmRange <- function(x) {
	return(x[x$model != "Nug",]$range[1])
}




gpuBenchmark <- function(nx = 100, ny = 100, k = 100, range=2,sill=5,nugget=0) {
	ITERATIONS = 5
	# build grid
	xmin = 0
	xmax = 5
	ymin = 0
	ymax = 5
	
	dx = (xmax-xmin)/nx
	dy = (ymax-ymin)/ny
	grid = GridTopology(c(xmin,ymin), c(dx,dy), c(nx,ny))


	cat("Generating 100 unconditional realizations at a 100x100 regular grid:\n")
	timeGPU = 0	
	for (z in 1:ITERATIONS) {
		try(timeGPU <- timeGPU + system.time(simGPU <- gpuSim(GridTopology(c(xmin,ymin), c(dx,dy), c(nx,ny)),"Exp", sill, range, nugget, k))[3])
	}
	timeGPU = timeGPU / ITERATIONS
	cat("Average computation time GPU: ")
	cat(timeGPU)
	cat("s\n")
		
	timeCPU = 0	
	for (z in 1:ITERATIONS) {
		temp = proc.time()[3]
		grid<- list( x= seq( xmin,xmax,,nx), y= seq(ymin,ymax,,ny)) 
		try(obj<-Exp.image.cov( grid=grid, theta=1/range, setup=TRUE))
		for (t in 1:k){
			try(sim.rf(obj))
		}
		timeCPU = timeCPU + (proc.time()[3] - temp)
	}
	timeCPU = timeCPU / ITERATIONS
	cat("Average computation time CPU using fields package: ")
	cat(timeCPU)
	cat("s\n")
	cat("Speedup factor: ")
	cat(timeCPU / timeGPU)
	cat("\n")
}

