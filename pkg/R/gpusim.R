

 gpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, as.sp = FALSE) {

	out = 0
 
	if (missing(grid)) {
		stop("Error: Missing grid argument!")
	}
	
	if (missing(covmodel) || missing(sill) || missing(range) || missing(nugget)) {
		stop("Error: Missing one or more arguments for covariance function!")
	}
	
	if (!missing(uncond) && !missing(samples)) {
		#only conditioning, k is ignored and derived from uncond object
		if (class(grid) != "GridTopology") {
			if (class(grid) == "SpatialPixelsDataFrame") {
				grid = grid@grid
			}
			else {
				stop("Error: grid must be of type SpatialPixelsDataFrame or GridTopology")
			}
		}
		
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
		
		if (class(uncond) != "array") {
			stop("Error: array expected as unconditional realizations!")
		}
		if (length(dim(uncond)) != 3) {
			stop("Error: expected a 3 dimensional array as unconditional realizations")
		}
		
		k = dim(uncond)[3]
		
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		

		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])
		
		## Get distance matrix -> cov matrix -> inverse cov matrix
		d <- spDists(samples,samples) ## Abstandsmatrix der Source Points untereinander
		cov <- covExponential(d, sill, range, nugget)	
			
		# Aus Abstandsmatrix->Kovarianzmatrix
		un = c(rep(1, numSrc),0)
		rn = rep(1, numSrc)
		cov.l = cbind(cov,rn)
		cov.l = rbind(cov.l,un)
		cov.l.inv = solve(cov.l)
		
		retcode = 0
		result = .C("conditioningInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc), as.single(cov.l.inv), as.integer(k), as.single(uncond), as.integer(covID(covmodel)), retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Initialization of conditioning returned error!")
		res = .C("conditioningRealizations", out=single(nx*ny*k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (res$retcode != 0) stop("Error: Generation of realizations returned error!")
		result = .C("conditioningRelease", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Release of simulation returned error")
		
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))			
		}
		
		
	}
	else if (!missing(k) && !missing(samples)) {
		#conditional simulation
		if (class(grid) != "GridTopology") {
			if (class(grid) == "SpatialPixelsDataFrame") {
				grid = grid@grid
			}
			else {
				stop("Error: grid must be of type SpatialPixelsDataFrame or GridTopology")
			}
		}
	
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
			
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		

		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])
		
		## Get distance matrix -> cov matrix -> inverse cov matrix
		d <- spDists(samples,samples) ## Abstandsmatrix der Source Points untereinander
		cov <- covExponential(d, sill, range, nugget)	#
		
		# Aus Abstandsmatrix->Kovarianzmatrix
		un = c(rep(1, numSrc),0)
		rn = rep(1, numSrc)
		cov.l = cbind(cov,rn)
		cov.l = rbind(cov.l,un)
		cov.l.inv = solve(cov.l)
					
		retcode = 0
		result = .C("conditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc), as.single(cov.l.inv), as.integer(covID(covmodel)),retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Initialization of conditional simulation returned error!")
		res = .C("conditionalSimRealizations", out=single(nx*ny*k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
		if (res$retcode != 0) stop("Error: Generation of realizations returned error!")
		result = .C("conditionalSimRelease",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop("Error: Release of simulation returned error")	

		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))			
		}		
		
		
		
	}
	else if (!missing(k)) {
		#uncond sim
		if (class(grid) != "GridTopology") {
			if (class(grid) == "SpatialPixelsDataFrame") {
				grid = grid@grid
			}
			else {
				stop("Error: grid must be of type SpatialPixelsDataFrame or GridTopology")
			}
		}

		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		
		
		retcode = 0
		result = .C("unconditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(covID(covmodel)), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Initialization of unconditional simulation returned error!")
		res = .C("unconditionalSimRealizations", out=single(nx*ny*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (res$retcode != 0) stop("Error: Generation of realizations returned error!")
		result = .C("unconditionalSimRelease", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Release of simulation returned error")	
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))			
		}
				
		
	}
	else {
		stop("Error: One or more required arguments missing!")
	}	
	
	return(out)	
}






gpuSimEval <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, as.sp = FALSE) {

	out = 0
	
	time.total = proc.time()[3]
	time.gpupre = 0
	time.gpureal = 0
 
	if (missing(grid)) {
		stop("Error: Missing grid argument!")
	}
	
	if (missing(covmodel) || missing(sill) || missing(range) || missing(nugget)) {
		stop("Error: Missing one or more arguments for covariance function!")
	}
	
	if (!missing(uncond) && !missing(samples)) {
		#only conditioning, k is ignored and derived from uncond object
		if (class(grid) != "GridTopology") {
			if (class(grid) == "SpatialPixelsDataFrame") {
				grid = grid@grid
			}
			else {
				stop("Error: grid must be of type SpatialPixelsDataFrame or GridTopology")
			}
		}
		
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
		
		if (class(uncond) != "array") {
			stop("Error: array expected as unconditional realizations!")
		}
		if (length(dim(uncond)) != 3) {
			stop("Error: expected a 3 dimensional array as unconditional realizations")
		}
		
		k = dim(uncond)[3]
		
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		

		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])
		
		## Get distance matrix -> cov matrix -> inverse cov matrix
		d <- spDists(samples,samples) ## Abstandsmatrix der Source Points untereinander
		cov <- covExponential(d, sill, range, nugget)
			
		# Aus Abstandsmatrix->Kovarianzmatrix
		un = c(rep(1, numSrc),0)
		rn = rep(1, numSrc)
		cov.l = cbind(cov,rn)
		cov.l = rbind(cov.l,un)
		cov.l.inv = solve(cov.l)
		
		retcode = 0
		time.gpupre = system.time(result <-.C("conditioningInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc), as.single(cov.l.inv), as.integer(k), as.single(uncond), as.integer(covID(covmodel)),retcode = as.integer(retcode),PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop("Error: Initialization of conditioning returned error!")	
		time.gpureal = system.time(res <- .C("conditioningRealizations", out=single(nx*ny*k), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (res$retcode != 0) stop("Error: Generation of realizations returned error!")
		result = .C("conditioningRelease",retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Release of simulation returned error")
		
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))			
		}
		
		
	}
	else if (!missing(k) && !missing(samples)) {
		#conditional simulation
		if (class(grid) != "GridTopology") {
			if (class(grid) == "SpatialPixelsDataFrame") {
				grid = grid@grid
			}
			else {
				stop("Error: grid must be of type SpatialPixelsDataFrame or GridTopology")
			}
		}
	
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
			
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		

		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])
		
		## Get distance matrix -> cov matrix -> inverse cov matrix
		d <- spDists(samples,samples) ## Abstandsmatrix der Source Points untereinander
		cov <- covExponential(d, sill, range, nugget)	
		
			
		# Aus Abstandsmatrix->Kovarianzmatrix
		un = c(rep(1, numSrc),0)
		rn = rep(1, numSrc)
		cov.l = cbind(cov,rn)
		cov.l = rbind(cov.l,un)
		cov.l.inv = solve(cov.l)
			
		retcode = 0
		time.gpupre = system.time(result <- .C("conditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc), as.single(cov.l.inv), as.integer(covID(covmodel)),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop("Error: Initialization of conditional simulation returned error!")	
		time.gpureal = system.time(res <- .C("conditionalSimRealizations", out=single(nx*ny*k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (res$retcode != 0) stop("Error: Generation of realizations returned error!")
		result <- .C("conditionalSimRelease",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop("Error: Release of simulation returned error")
		
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))			
		}		
		
		
		
	}
	else if (!missing(k)) {
		#uncond sim
		if (class(grid) != "GridTopology") {
			if (class(grid) == "SpatialPixelsDataFrame") {
				grid = grid@grid
			}
			else {
				stop("Error: grid must be of type SpatialPixelsDataFrame or GridTopology")
			}
		}

		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		
		retcode = 0
		time.gpupre = system.time(result <- .C("unconditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(covID(covmodel)),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop("Error: Initialization of unconditional simulation returned error!")	
		time.gpureal = system.time(res <- .C("unconditionalSimRealizations", out=single(nx*ny*k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (res$retcode != 0) stop("Error: Generation of realizations returned error!")
		result <- .C("unconditionalSimRelease", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop("Error: Release of simulation returned error")
		
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))			
		}
				
		
	}
	else {
		stop("Error: One or more required arguments missing!")
	}	
	
	time.total = proc.time()[3] - time.total
	time.remaining = time.total - time.gpupre - time.gpureal
	
	return(c(time.total = time.total, time.gpupreprocessing= time.gpupre, time.gpurealizations = time.gpureal, time.cpuremaining = time.remaining))
}








## IDs must be equal to enumeration in C code
covID <- function(covmodel) {
	if (covmodel == "Exp") return(0)
	else if (covmodel == "Sph") return(1)
	else if (covmodel == "Gau") return(2)
	else stop("Unknown covariance function!")
}



covExponential <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covExp", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)	
}

covGaussian <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covGau", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)
}

covSpherical <- function(data, sill, range, nugget) {	
	n = length(data)
	res = .C("covSph", out=numeric(n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), PACKAGE="gpusim")
	dim(res$out) = dim(data)
	return(res$out)
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

