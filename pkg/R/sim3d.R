 
 
 
.sim3d <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE, anis=c(0,0,0,1,1)) {
	out = 0
 
	if (!missing(uncond) && !missing(samples)) {
		#only conditioning, k is ignored and derived from uncond object
		
		##########################################
		stop("NOT YET IMPLEMENTED!")
		##########################################
		
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
		
		if (class(uncond) != "array") {
			stop("Error: array expected as unconditional realizations!")
		}
		if (length(dim(uncond)) != 3) {
			stop("Error: expected a 3 dimensional array as unconditional realizations")
		}
	
	
		if (all(c('O','o','S','s') != kriging.method)) {
			stop("Error: Unknown kriging method")
		}
		if (any(c('s','S') == kriging.method) && missing(mu)) {
			mu = 0
			warning("No mean for simple kriging given, using 0")
		}
		
		
		
		k = dim(uncond)[3]
		
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		zmin = grid@cellcentre.offset[3]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		dz = grid@cellsize[3]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		nz = grid@cells.dim[3]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		zmax = zmin + nz * dz		
		

		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])	
		cov.l <- dCov3d(coordinates(samples),covmodel,sill,range,nugget,anis)
	
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res = 0
		retcode = 0
		result = .C("conditioningInit_3d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.double(srcXY), as.double(srcData), as.integer(numSrc),  as.integer(k), as.double(uncond), as.integer(.covID(covmodel)), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditioning returned error:",.gpuSimCatchError(result$retcode)))
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditioningResiduals_3d", out=double((numSrc + 1) * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditioningKrigeResiduals_3d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(res$retcode)))
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditioningResiduals_3d", out=double(numSrc * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditioningSimpleKrigeResiduals_3d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
		}		
		
		result = .C("conditioningRelease_3d", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for simulation returned error:" , .gpuSimCatchError(result$retcode)))
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny*nz)))	
			names(out@data) = paste("sim",1:k,sep="")
		}
			
	}
	else if (!missing(k) && !missing(samples)) {
		#conditional simulation
		
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
		
		if (all(c('O','o','S','s') != kriging.method)) {
			stop("Error: Unknown kriging method")
		}
		if (any(c('s','S') == kriging.method) && missing(mu)) {
			mu = 0
			warning("No mean for simple kriging given, using 0")
		}
	
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		zmin = grid@cellcentre.offset[3]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		dz = grid@cellsize[3]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		nz = grid@cells.dim[3]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		zmax = zmin + nz * dz	
	
		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 3
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])
		
		# Get covariance matrix from sample points
		cov.l <- dCov3d(coordinates(samples),covmodel,sill,range,nugget,anis)	
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res <- 0		
		retcode = 0
		result = .C("conditionalSimInit_3d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(zmin), as.double(zmax), as.integer(nz), as.double(sill), as.double(range), as.double(nugget), as.double(srcXY), as.double(srcData), as.integer(numSrc), as.integer(.covID(covmodel)), as.double(anis), as.integer(check), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditionalSimUncondResiduals_3d", out=double((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditionalSimKrigeResiduals_3d", out=double(nx*ny*nz*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(res$retcode)))
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditionalSimUncondResiduals_3d", out=double(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditionalSimSimpleKrigeResiduals_3d", out=double(nx*ny*nz*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
		}		
		
		# clean up
		result = .C("conditionalSimRelease_3d",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			

		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,nz,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny*nz)))
			names(out@data) = paste("sim",1:k,sep="")
		}				
	}
	else if (!missing(k)) {
		#uncond sim
		
		
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		zmin = grid@cellcentre.offset[3]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		dz = grid@cellsize[3]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		nz = grid@cells.dim[3]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
		zmax = zmin + nz * dz
		
		
		retcode = 0
		result = .C("unconditionalSimInit_3d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(zmin), as.double(zmax), as.integer(nz), as.double(sill), as.double(range), as.double(nugget), as.integer(.covID(covmodel)), as.double(anis), as.integer(check), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))			
		res = .C("unconditionalSimRealizations_3d", out=double(nx*ny*nz*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		result = .C("unconditionalSimRelease_3d", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
				
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,nz,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny*nz)))
			names(out@data) = paste("sim",1:k,sep="")
		}		
	}
	else {
		stop("Error: Missing one or more required arguments!")
	}

	if (verify) {
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

	return(out)	
}





.sim3dEval <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE) {
	
	##########################################
	stop("NOT YET IMPLEMENTED!")
	##########################################
	
	time.total <- proc.time()[3] # total runtime of function
	times = c() # runtimes of double computation steps
	out = 0
 
	
	if (!missing(uncond) && !missing(samples)) {
		#only conditioning, k is ignored and derived from uncond object
		
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
		
		if (class(uncond) != "array") {
			stop("Error: array expected as unconditional realizations!")
		}
		if (length(dim(uncond)) != 3) {
			stop("Error: expected a 3 dimensional array as unconditional realizations")
		}
	
	
		if (all(c('O','o','S','s') != kriging.method)) {
			stop("Error: Unknown kriging method")
		}
		if (any(c('s','S') == kriging.method) && missing(mu)) {
			mu = 0
			warning("No mean for simple kriging given, using 0")
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
		

		t1 = proc.time()[3]
		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])	
		cov.l <- dCov2d(coordinates(samples),covmodel,sill,range,nugget)
		t1 = proc.time()[3] - t1
		names(t1) = "CPU Preprocessing Input Samples"
		times = c(times,t1)
	
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res = 0
		retcode = 0
		t1 = system.time(result <- .C("conditioningInit_2d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.double(srcXY), as.double(srcData), as.integer(numSrc),  as.integer(k), as.double(uncond), as.integer(.covID(covmodel)), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), retcode = as.integer(retcode),PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop(paste("Initialization of conditioning returned error:",.gpuSimCatchError(result$retcode)))	
		names(t1) = "GPU Initialization"
		times = c(times,t1)
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditioningResiduals_2d", out=double((numSrc + 1) * k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Residual Computation"
			times = c(times,t1)
			
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			t1 = system.time(y <- solve(cov.l, res$out))[3]
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)			
			# interpolate residuals and add to the unconditional realizations		
			t1  = system.time(res <- .C("conditioningKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditioningResiduals_2d", out=double(numSrc * k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Residual Computation"
			times = c(times,t1)
			
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			t1 <- system.time(y = solve(cov.l, res$out))[3]
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)
			
			# interpolate residuals and add to the unconditional realizations		
			t1 <- system.time(res <- .C("conditioningSimpleKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}		
		
		result = .C("conditioningRelease_2d", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for simulation returned error:" , .gpuSimCatchError(result$retcode)))
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny*nz)))
			names(out@data) = paste("sim",1:k,sep="")
		}
			
	}
	else if (!missing(k) && !missing(samples)) {
		#conditional simulation
	
		if (class(samples) != "SpatialPointsDataFrame") {
			stop("Error: samples must be of type SpatialPointsDataFrame")
		}
		
		if (all(c('O','o','S','s') != kriging.method)) {
			stop("Error: Unknown kriging method")
		}
		if (any(c('s','S') == kriging.method) && missing(mu)) {
			mu = 0
			warning("No mean for simple kriging given, using 0")
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
		
		t1 = proc.time()[3]
		srcData <- as.vector(samples@data[,1])	
		# Get covariance matrix from sample points
		cov.l <- dCov2d(coordinates(samples),covmodel,sill,range,nugget)	
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		t1 = proc.time()[3] - t1
		names(t1) = "CPU Preprocessing Input Samples"
		times = c(times,t1)
		
		
		res <- 0		
		retcode = 0
		result <- .C("conditionalSimInit_2d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.double(srcXY), as.double(srcData), as.integer(numSrc), as.integer(.covID(covmodel)), as.integer(check), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditionalSimUncondResiduals_2d", out=double((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Generate unconditional Realizations and compute Residuals"
			times = c(times,t1)	
			
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			t1 <- system.time(y <- solve(cov.l, res$out))[3]				
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)	
			
			# interpolate residuals and add to the unconditional realizations		
			t1 = system.time(res <- .C("conditionalSimKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditionalSimUncondResiduals_2d", out=double(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Generate unconditional Realizations and compute Residuals"
			times = c(times,t1)
			
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			t1 = system.time(y <- solve(cov.l, res$out))[3]	
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)
			
			# interpolate residuals and add to the unconditional realizations		
			t1 = system.time(res <- .C("conditionalSimSimpleKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}		
		
		# clean up
		result = .C("conditionalSimRelease_2d",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			

		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny*nz)))
			names(out@data) = paste("sim",1:k,sep="")
		}				
	}
	else if (!missing(k)) {
		#uncond sim
	
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + nx * dx
		ymax = ymin + ny * dy	
			
		retcode = 0
		t1 = system.time(result <- .C("unconditionalSimInit_2d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.integer(.covID(covmodel)), as.integer(check), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))			
		names(t1) = "GPU Initialization"
		times = c(times,t1)
		
		t1 = system.time(res <- .C("unconditionalSimRealizations_2d", out=double(nx*ny*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		names(t1) = "GPU Generatiion of Unconditional Realizations"
		times = c(times,t1)
			
		result = .C("unconditionalSimRelease_2d", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
				
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,nz,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny*nz)))	
			names(out@data) = paste("sim",1:k,sep="")
		}		
	}
	else {
		stop("Error: Missing one or more required arguments!")
	}

	if (verify) {
		t1 <- proc.time()[3]
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
		
		t1 <- proc.time()[3] - t1
		names(t1) = "CPU Verification"
		times = c(times,t1)
	}
	
	
	
	time.total <- proc.time()[3] - time.total
	names(time.total) = "Total"
	time.remaining = time.total - sum(times)
	names(time.remaining) = "Remaining"
	times = c(time.total, times, time.remaining)
	retval = list(result = out, runtimes = times)
	return(retval)	
}












