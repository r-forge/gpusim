 
 
 gpuDeviceInfo <- function() {
	cat("\n******** GPU DEVICE INFO ********\n")
	result = .C("deviceInfo", out=format("",width=512),PACKAGE="gpusim")
	cat(result$out)	
	cat("*********************************\n\n")
 }

 gpuReset <- function() {
	retcode = 0
	result = .C("reset",retcode = as.integer(retcode),PACKAGE="gpusim")
	if (result$retcode != 0) stop(paste("GPU device reset returned error:",.gpuSimCatchError(result$retcode)))	
	cat("\n******** GPU DEVICE HAS BEEN RESET ********\n")
	gc()
 }


gpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method='O', mu=0, aggregation.features=NULL, aggregation.func=mean, gpu.cache=TRUE, as.sp=FALSE, neg.eigenvals.action = "ignore", eigenvals.tol=-1e-07,  benchmark=FALSE, prec.double=FALSE, compute.stats=FALSE, anis=c(0,0,0,1,1), cpu.invertonly=FALSE) {
	
	if (missing(grid)) {
		stop("Error: Missing grid argument!")
	}
	
	if (missing(covmodel) || missing(sill) || missing(range) || missing(nugget)) {
		stop("Error: Missing one or more arguments for covariance function!")
	}
	
	if (class(grid) != "GridTopology") {
		if (class(grid) == "SpatialPixelsDataFrame" || class(grid) == "SpatialGridDataFrame") {
			grid = grid@grid
		}
		else {
			stop("Error: grid must be of type SpatialPixelsDataFrame, SpatialGridDataFrame, or GridTopology")
		}
	}
	
	# always use 5 pars to describe anisotropy
	if (length(anis) == 2) {
		anis = c(anis[1],0,0,anis[2],1)
	}
	if (length(anis) != 5) {
		stop("Expected 5 or 2 anisotropy values!")
	}

	
	
	# aggregation input args check
	if (!is.null(aggregation.features) && !is.null(aggregation.func)) {
		if (!as.sp) {
			warning("Notice that aggregation forces as.sp = TRUE")
			as.sp = TRUE
		}
		if (class(aggregation.features) == "SpatialPolygonsDataFrame") {
			aggregation.features = SpatialPolygons(aggregation.features@polygons)
		}
		if (class(aggregation.features) != "SpatialPolygons") {
			stop("Feature aggregation requires polygon features as SpatialPolygons")
		}
	}

	dims = length(grid@cells.dim)
	out <- 0
	if (dims == 2) {
		if (prec.double) {
			out <- .sim2d(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, aggregation.features, aggregation.func, gpu.cache, as.sp, neg.eigenvals.action, eigenvals.tol, benchmark, compute.stats, anis, cpu.invertonly)
			if (neg.eigenvals.action == "output") {
				xmin = grid@cellcentre.offset[1]
				ymin = grid@cellcentre.offset[2]
				dx = grid@cellsize[1]
				dy = grid@cellsize[2]
				nx = grid@cells.dim[1]
				ny = grid@cells.dim[2]
				xmax = xmin + (nx-1) * dx
				ymax = ymin + (ny-1) * dy
				out = list(sim = out)
				result = .C("simEigenVals_2d", out = double(prod(grid@cells.dim)*4), as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.integer(.covID(covmodel)), as.double(anis[1]), as.double(anis[4]),as.double(eigenvals.tol), PACKAGE="gpusim")	
				dim(result$out) = 2*grid@cells.dim
				out$eigvals = result$out
				message("Result of simulation is a list of two elements, the simulation result as well as the computed eigenvalues!")
			}
		}
		else {
			out <- .sim2f(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, aggregation.features, aggregation.func, gpu.cache, as.sp, neg.eigenvals.action, eigenvals.tol,benchmark, compute.stats, anis, cpu.invertonly)
			if (neg.eigenvals.action == "output") {
				xmin = grid@cellcentre.offset[1]
				ymin = grid@cellcentre.offset[2]
				dx = grid@cellsize[1]
				dy = grid@cellsize[2]
				nx = grid@cells.dim[1]
				ny = grid@cells.dim[2]
				xmax = xmin + (nx-1) * dx
				ymax = ymin + (ny-1) * dy
				out = list(sim = out)
				result = .C("simEigenVals_2f", out = single(prod(grid@cells.dim)*4), as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(.covID(covmodel)), as.single(anis[1]), as.single(anis[4]),as.single(eigenvals.tol), PACKAGE="gpusim")	
				dim(result$out) = 2*grid@cells.dim
				out$eigvals = result$out
				message("Result of simulation is a list of two elements, the simulation result as well as the computed eigenvalues!")
			}
		}
	}
	else if (dims == 3) {
		if (cpu.invertonly) warning("cpu.invertonly is only used for testing purposes in two-dimensional simulation, argument will be ignored...")
		if (prec.double) {
			out <- .sim3d(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, aggregation.features, aggregation.func, gpu.cache, as.sp, check=FALSE, benchmark, compute.stats, anis, cpu.invertonly)
			if (neg.eigenvals.action == "output") {
				message("Three-dimensional eigenvalue computation currently not implemented, will be ignored!")
				#out = list(sim = out)
				#TODO: ADD z grid definition... result = .C("simEigenVals_3d", out = double(prod(grid@cells.dim)*8), as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(.covID(covmodel)), as.single(anis[1]), as.single(anis[4]),as.single(eigenvals.tol), PACKAGE="gpusim")	
				#dim(result$out) = 2*grid@cells.dim
				#out$eigvals = result$out
				#message("Result of simulation is a list of two elements, the simulation result as well as the computed eigenvalues!")
			}
		}
		else {
			out <- .sim3f(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, aggregation.features, aggregation.func, gpu.cache, as.sp, check=FALSE, benchmark, compute.stats, anis, cpu.invertonly)
			if (neg.eigenvals.action == "output") {
				message("Three-dimensional eigenvalue computation currently not implemented, will be ignored!")
				#out = list(sim = out)
				#TODO: ADD z grid definition... result = .C("simEigenVals_3f", out = double(prod(grid@cells.dim)*8), as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(.covID(covmodel)), as.single(anis[1]), as.single(anis[4]),as.single(eigenvals.tol), PACKAGE="gpusim")	
				#dim(result$out) = 2*grid@cells.dim
				#out$eigvals = result$out
				#message("Result of simulation is a list of two elements, the simulation result as well as the computed eigenvalues!")
			}
		}
	}
	else stop("Only two- or three-dimensional simulation supported!")
	
	
	
	return(out)
}












gpuSimEval <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE, prec.double=FALSE) {
	stop("Evaluation function currently not working but will be available soon")
	
	if (missing(grid)) {
		stop("Error: Missing grid argument!")
	}
	
	if (missing(covmodel) || missing(sill) || missing(range) || missing(nugget)) {
		stop("Error: Missing one or more arguments for covariance function!")
	}
		
	if (verify == TRUE && as.sp == FALSE) {
		cat("Notice that verification forces as.sp = TRUE!\n")
		as.sp = TRUE
	}
	
	if (class(grid) != "GridTopology") {
		if (class(grid) == "SpatialPixelsDataFrame" || class(grid) == "SpatialGridDataFrame") {
			grid = grid@grid
		}
		else {
			stop("Error: grid must be of type SpatialPixelsDataFrame, SpatialGridDataFrame, or GridTopology")
		}
	}
	
	# always use 5 pars to describe anisotropy
	if (length(anis) == 2) {
		anis = c(anis[1],0,0,anis[2],1)
	}
	if (length(anis) != 5) {
		stop("Expected 5 or 2 anisotropy values!")
	}
	
	dims = length(grid@cells.dim)
	if (dims == 2) {
		if (prec.double) {
			return(.sim2dEval(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify, anis))
		}
		else return(.sim2fEval(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify, anis))
	}
	else if (dims == 3) {
		if (prec.double) {
			return(.sim3dEval(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, as.sp, check, verify, anis))
		}
		else stop("3d simulation in single precision currently not implemented, use prec.double = TRUE instead!")
	}
	else stop("Only two- or three-dimensional simulation supported!")
}



gpuSimVerify <- function(simsp) {

	if (class(sim) != "SpatialGridDataFrame") {
		stop("Verification needs input type SpatialGridDataFrame. Use as.sp = TRUE for simulation before.")
	}

	gamma = 0
	v0 = variogram(out[[1]]~1,out,cloud=FALSE)
	for (i in 2:ncol(out)) {
		v = variogram(out[[i]]~1,out,cloud=FALSE)
		v0$gamma = v0$gamma + v$gamma
	}
	v0$gamma = v0$gamma / ncol(out)
	print(plot(v0,vgm(sill, model, range, nugget)))	
	cat("Verification returned the following experimental variogram: \n")
	print(v0)
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



gpuCov2 <- function(data, model, sill, range, nugget, anis=c(0,0,0,1,1), prec.double=FALSE) {
	if (class(data) != "matrix") {
		stop("Expected a matrix as input!")
	}
	n = nrow(data)
	if (ncol(data) != 2) {
		stop("Expected a matrix with exactly 2 columns!")
	}
	res = 0
	if (anis[4] == 1) {
		if (prec.double) {
			res = .C("gpuCov_2d", out=double(n*n), as.double(data), as.integer(n), as.integer(.covID(model)), as.double(sill), as.double(range), as.double(nugget), PACKAGE="gpusim")
		}
		else {
			res = .C("gpuCov_2f", out=single(n*n), as.single(data), as.integer(n), as.integer(.covID(model)), as.single(sill), as.single(range), as.single(nugget), PACKAGE="gpusim")
		}
	}
	else { #anisotropic
		if (prec.double) {
			res = .C("gpuCovAnis_2d", out=double(n*n), as.double(data), as.integer(n), as.integer(.covID(model)),as.double(sill), as.double(range), as.double(nugget), as.double(anis[1]), as.double(anis[4]), PACKAGE="gpusim")
		}
		else {
			res = .C("gpuCovAnis_2f", out=single(n*n), as.single(data), as.integer(n), as.integer(.covID(model)),as.single(sill), as.single(range), as.single(nugget), as.single(anis[1]), as.single(anis[4]), PACKAGE="gpusim")		
		}
	}	
	dim(res$out) = c(n,n)
	return(res$out)
}


dCov2d <- function(data, model, sill, range, nugget, anis=c(0,0,0,1,1)) {
	if (class(data) != "matrix") {
		stop("Expected a matrix as input!")
	}
	n = nrow(data)
	if (ncol(data) != 2) {
		stop("Expected a matrix with exactly 2 columns!")
	}
	res = 0
	if (anis[4] == 1) {
		res = .C("dCov2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.integer(.covID(model)), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	}
	else { #anisotropic
		res = .C("dCovAnis2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.integer(.covID(model)),as.numeric(sill), as.numeric(range), as.numeric(nugget), as.numeric(anis[1]), as.numeric(anis[4]), DUP=TRUE, PACKAGE="gpusim")		
	}	
	dim(res$out) = c(n,n)
	return(res$out)
}





dCov3d <- function(data, model, sill, range, nugget, anis=c(0,0,0,1,1)) {
	if (class(data) != "matrix") {
		stop("Expected a matrix as input!")
	}
	n = nrow(data)
	if (ncol(data) != 3) {
		stop("Expected a matrix with exactly 3 columns!")
	}
	res = 0
	if (anis[4] == 1 && anis[5] == 1) {
		res = .C("dCov3d", out=numeric(n*n), as.numeric(data), as.integer(n), as.integer(.covID(model)), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	}
	else { #anisotropic
		res = .C("dCovAnis3d", out=numeric(n*n), as.numeric(data), as.integer(n), as.integer(.covID(model)),as.numeric(sill), as.numeric(range), as.numeric(nugget), as.numeric(anis[1]), as.numeric(anis[2]), as.numeric(anis[3]), as.numeric(anis[4]), as.numeric(anis[5]), DUP=TRUE, PACKAGE="gpusim")		
	}	
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

## Gets the 5 anisotropy values of the model
vgmAnis <- function(x) {
	return(c(x$ang1,x$ang2,x$ang3,x$anis1,x$anis2))
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

