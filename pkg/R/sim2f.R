 
 
 
.sim2f <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, as.sp = FALSE, check = FALSE, benchmark = FALSE, anis=c(0,0,0,1,1)) {
	
	if (benchmark) {
		times = c() # runtimes of single computation steps
		.gpuSimStartTimer()
	}
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
		

		if (benchmark) .gpuSimStartTimer()
		
		numSrc = length(samples)	
		
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])	
		cov.l <- dCov2d(coordinates(samples),covmodel,sill,range,nugget,anis)
		
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "CPU Preprocessing Input Samples"
			times = c(times,t1)
		}
	
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res = 0
		retcode = 0
		if (benchmark) .gpuSimStartTimer()
		
		result = .C("conditioningInit_2f", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(t(coordinates(samples))), as.single(srcData), as.integer(numSrc),  as.integer(k), as.single(uncond), as.integer(.covID(covmodel)), as.single(anis[1]), as.single(anis[4]), as.integer(.gpuSimKrigeMethod(kriging.method)), as.single(mu), retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditioning returned error:",.gpuSimCatchError(result$retcode)))
		
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Initialization"
			times = c(times,t1)
		}
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			if (benchmark) .gpuSimStartTimer()
			res = .C("conditioningResiduals_2f", out=single((numSrc + 1) * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
			
			if (benchmark) .gpuSimStartTimer()
			
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			y = solve(cov.l, res$out)				
			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations
			if (benchmark) .gpuSimStartTimer()			
			res = .C("conditioningKrigeResiduals_2f", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(res$retcode)))
			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Ordinary Kriging Residuals"
				times = c(times,t1)
			}
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			if (benchmark) .gpuSimStartTimer()
			res = .C("conditioningResiduals_2f", out=single(numSrc * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
					
			# solve residual equation system
			if (benchmark) .gpuSimStartTimer()	
			dim(res$out) = c(numSrc,k)
			y = solve(cov.l, res$out)				
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations	
			if (benchmark) .gpuSimStartTimer()	
			res = .C("conditioningSimpleKrigeResiduals_2f", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Simple Kriging Residuals"
				times = c(times,t1)
			}
		}		
		
		result = .C("conditioningRelease_2f", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for simulation returned error:" , .gpuSimCatchError(result$retcode)))
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))		
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
	
		if (benchmark) .gpuSimStartTimer()
		numSrc = length(samples)	
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])	
		# Get covariance matrix from sample points
		cov.l <- dCov2d(coordinates(samples),covmodel,sill,range,nugget,anis)
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "CPU Preprocessing Input Samples"
			times = c(times,t1)
		}
		
		if (any(c('O','o') == kriging.method)) {
			cov.l = cbind(cov.l,rep(1, numSrc))
			cov.l = rbind(cov.l,c(rep(1, numSrc),0))
		}
		
		res <- 0		
		retcode = 0
		if (benchmark) .gpuSimStartTimer()
		result = .C("conditionalSimInit_2f", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.single(t(coordinates(samples))), as.single(srcData), as.integer(numSrc), as.integer(.covID(covmodel)), as.single(anis[1]), as.single(anis[4]), as.integer(check), as.integer(.gpuSimKrigeMethod(kriging.method)), as.single(mu), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Initialization"
			times = c(times,t1)
		}
		
		# if ordinary kriging add lagrange condition
		if (any(c('O','o') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			if (benchmark) .gpuSimStartTimer()	
			res = .C("conditionalSimUncondResiduals_2f", out=single((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
			
			# solve residual equation system
			if (benchmark) .gpuSimStartTimer()	
			dim(res$out) = c(numSrc+1,k)
			y = solve(cov.l, res$out)				
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations		
			if (benchmark) .gpuSimStartTimer()	
			res = .C("conditionalSimKrigeResiduals_2f", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(res$retcode)))
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Ordinary Kriging Residuals"
				times = c(times,t1)
			}
		}
		else if (any(c('S','s') == kriging.method)) {	
			# generate all unconditional realizations and get their residuals to the data
			if (benchmark) .gpuSimStartTimer()	
			res = .C("conditionalSimUncondResiduals_2f", out=single(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
			
			# solve residual equation system
			if (benchmark) .gpuSimStartTimer()
			dim(res$out) = c(numSrc,k)
			y = solve(cov.l, res$out)			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations		
			if (benchmark) .gpuSimStartTimer()
			res = .C("conditionalSimSimpleKrigeResiduals_2f", out=single(nx*ny*k), as.single(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Simple Kriging Residuals"
				times = c(times,t1)
			}
		}		
		
		# clean up
		result = .C("conditionalSimRelease_2f",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			

		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))	
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
		
		if (benchmark) .gpuSimStartTimer()
		result = .C("unconditionalSimInit_2f", as.single(xmin), as.single(xmax), as.integer(nx), as.single(ymin),as.single(ymax), as.integer(ny), as.single(sill), as.single(range), as.single(nugget), as.integer(.covID(covmodel)), as.single(anis[1]), as.single(anis[4]), as.integer(check), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))			
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Initialization"
			times = c(times,t1)
		}
		
		if (benchmark) .gpuSimStartTimer()
		res = .C("unconditionalSimRealizations_2f", out=single(nx*ny*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Generating Unconditional Realizations"
			times = c(times,t1)
		}
	
		result = .C("unconditionalSimRelease_2f", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			
		if (as.sp == FALSE) {
			out = as.vector(res$out)
			dim(out) = c(nx,ny,k)
		}
		else {
			out = SpatialGridDataFrame(grid,as.data.frame(matrix(res$out,ncol = k,nrow = nx*ny)))	
			names(out@data) = paste("sim",1:k,sep="")
		}		
	}
	else {
		stop("Error: Missing one or more required arguments!")
	}

	if (benchmark) {
		time.total <- .gpuSimStopTimer()
		names(time.total) = "Total"
		time.remaining = time.total - sum(times)
		names(time.remaining) = "Remaining"
		times = c(time.total, times, time.remaining)
		.gpuSimStopTimer(T)	
		print(times)
	}
	return(out)	
}




