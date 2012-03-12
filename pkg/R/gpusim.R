 
 
 gpuDeviceInfo <- function() {
	cat("\n******** GPU DEVICE INFO ********\n")
	result = .C("deviceInfo", out=format("",width=255),PACKAGE="gpusim")
	cat(result$out)	
	cat("*********************************\n\n")
 }
 
 gpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE) {
	out = 0
 
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
		

		srcXY  <- as.vector(t(coordinates(samples)))
		numSrc = length(srcXY) / 2
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])	
		cov.l <- dCov2d(coordinates(samples),covmodel,sill,range,nugget)
	
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res = 0
		retcode = 0
		result = .C("conditioningInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc),  as.integer(k), as.single(uncond), as.integer(covID(covmodel)), as.integer(gpuSimKrigeMethod(kriging.method)), as.single(mu), retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditioning returned error:",gpuSimCatchError(result$retcode)))
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditioningResiduals", out=single((numSrc + 1) * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditioningKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",gpuSimCatchError(res$retcode)))
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditioningResiduals", out=single(numSrc * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditioningSimpleKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", gpuSimCatchError(res$retcode)))
		}		
		
		result = .C("conditioningRelease", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for simulation returned error:" , gpuSimCatchError(result$retcode)))
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))	
			names(out@data) = paste("sim",1:k,sep="")
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
		srcData <- as.vector(samples@data[,1])
		
		# Get covariance matrix from sample points
		cov.l <- dCov2d(coordinates(samples),covmodel,sill,range,nugget)
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res <- 0		
		retcode = 0
		result = .C("conditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc), as.integer(covID(covmodel)), as.integer(check), as.integer(gpuSimKrigeMethod(kriging.method)), as.single(mu), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditional simulation returned error: ",gpuSimCatchError(result$retcode)))
		
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditionalSimUncondResiduals", out=single((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditionalSimKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",gpuSimCatchError(res$retcode)))
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			res = .C("conditionalSimUncondResiduals", out=single(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			y = solve(cov.l, res$out)				
			# interpolate residuals and add to the unconditional realizations		
			res = .C("conditionalSimSimpleKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", gpuSimCatchError(res$retcode)))
		}		
		
		# clean up
		result = .C("conditionalSimRelease",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",gpuSimCatchError(result$retcode)))
			

		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))
			names(out@data) = paste("sim",1:k,sep="")
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
		result = .C("unconditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(covID(covmodel)), as.integer(check), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",gpuSimCatchError(result$retcode)))			
		res = .C("unconditionalSimRealizations", out=single(nx*ny*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",gpuSimCatchError(result$retcode)))
		result = .C("unconditionalSimRelease", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",gpuSimCatchError(result$retcode)))
				
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))	
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



















gpuSimEval <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, verify = FALSE) {
	time.total <- proc.time()[3] # total runtime of function
	times = c() # runtimes of single computation steps
	out = 0
 
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
		t1 = system.time(result <- .C("conditioningInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc),  as.integer(k), as.single(uncond), as.integer(covID(covmodel)), as.integer(gpuSimKrigeMethod(kriging.method)), as.single(mu), retcode = as.integer(retcode),PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop(paste("Initialization of conditioning returned error:",gpuSimCatchError(result$retcode)))	
		names(t1) = "GPU Initialization"
		times = c(times,t1)
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditioningResiduals", out=single((numSrc + 1) * k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Residual Computation"
			times = c(times,t1)
			
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			t1 = system.time(y <- solve(cov.l, res$out))[3]
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)			
			# interpolate residuals and add to the unconditional realizations		
			t1  = system.time(res <- .C("conditioningKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditioningResiduals", out=single(numSrc * k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Residual Computation"
			times = c(times,t1)
			
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			t1 <- system.time(y = solve(cov.l, res$out))[3]
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)
			
			# interpolate residuals and add to the unconditional realizations		
			t1 <- system.time(res <- .C("conditioningSimpleKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}		
		
		result = .C("conditioningRelease", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for simulation returned error:" , gpuSimCatchError(result$retcode)))
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))	
			names(out@data) = paste("sim",1:k,sep="")
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
		result <- .C("conditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(srcXY), as.single(srcData), as.integer(numSrc), as.integer(covID(covmodel)), as.integer(check), as.integer(gpuSimKrigeMethod(kriging.method)), as.single(mu), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditional simulation returned error: ",gpuSimCatchError(result$retcode)))
		
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditionalSimUncondResiduals", out=single((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Generate unconditional Realizations and compute Residuals"
			times = c(times,t1)	
			
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			t1 <- system.time(y <- solve(cov.l, res$out))[3]				
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)	
			
			# interpolate residuals and add to the unconditional realizations		
			t1 = system.time(res <- .C("conditionalSimKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			t1 = system.time(res <- .C("conditionalSimUncondResiduals", out=single(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",gpuSimCatchError(res$retcode)))		
			names(t1) = "GPU Generate unconditional Realizations and compute Residuals"
			times = c(times,t1)
			
			# solve residual equation system
			dim(res$out) = c(numSrc,k)
			t1 = system.time(y <- solve(cov.l, res$out))[3]	
			names(t1) = "CPU Solving Equation System"
			times = c(times,t1)
			
			# interpolate residuals and add to the unconditional realizations		
			t1 = system.time(res <- .C("conditionalSimSimpleKrigeResiduals", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", gpuSimCatchError(res$retcode)))
			names(t1) = "GPU Residual Kriging"
			times = c(times,t1)
		}		
		
		# clean up
		result = .C("conditionalSimRelease",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",gpuSimCatchError(result$retcode)))
			

		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))
			names(out@data) = paste("sim",1:k,sep="")
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
		t1 = system.time(result <- .C("unconditionalSimInit", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(covID(covmodel)), as.integer(check), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",gpuSimCatchError(result$retcode)))			
		names(t1) = "GPU Initialization"
		times = c(times,t1)
		
		t1 = system.time(res <- .C("unconditionalSimRealizations", out=single(nx*ny*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim"))[3]
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",gpuSimCatchError(result$retcode)))
		names(t1) = "GPU Generatiion of Unconditional Realizations"
		times = c(times,t1)
			
		result = .C("unconditionalSimRelease", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",gpuSimCatchError(result$retcode)))
				
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialPixelsDataFrame(coordinates(grid),as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))	
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












## IDs must be equal to enumeration in C code
covID <- function(covmodel) {
	if (covmodel == "Exp") return(0)
	else if (covmodel == "Sph") return(1)
	else if (covmodel == "Gau") return(2)
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


dCov2d <- function(data, model, sill, range, nugget) {
	if (class(data) != "matrix") {
		stop("Expected a matrix as input!")
	}
	n = nrow(data)
	if (ncol(data) != 2) {
		stop("Expected a matrix with exactly 2 columns!")
	}
	res = 0
	if (model == "Exp") res = .C("dCovExp2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	else if (model == "Gau") res = .C("dCovGau2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")	
	else if (model == "Sph") res = .C("dCovSph2d", out=numeric(n*n), as.numeric(data), as.integer(n), as.numeric(sill), as.numeric(range), as.numeric(nugget), DUP=FALSE, PACKAGE="gpusim")
	else stop("Unknown covariance model!")	
	dim(res$out) = c(n,n)
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

