 
 
 
 
 
 
 .sim3d <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'S', mu = 0, aggregation.features=NULL, aggregation.func=mean, gpu.cache = FALSE, as.sp = FALSE, check = FALSE, benchmark = FALSE, compute.stats = FALSE, anis=c(0,0,0,1,1)) {
	
	if (benchmark) {
		times = c() # runtimes of single computation steps
		.gpuSimStartTimer()
	}
	out = 0
	
	if (!missing(uncond) && !missing(samples)) {
		#only conditioning, k is ignored and derived from uncond object
		
		
		##########################################
		stop("NOT YET IMPLEMENTED!")
		##########################################
		
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
		xmax = xmin + (nx-1) * dx
		ymax = ymin + (ny-1) * dy	
		zmax = zmin + (nz-1) * dz		
	
		if (benchmark) .gpuSimStartTimer()
		numSrc = length(samples)	
		if (length(samples@data) != 1) {
			stop("Error: samples contain more than one data field!")
		}
		srcData <- as.vector(samples@data[,1])	
		# Get covariance matrix from sample points
		cov.l <- dCov3d(coordinates(samples),covmodel,sill,range,nugget,anis)
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
		result = .C("conditionalSimInit_3d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(zmin), as.double(zmax), as.integer(nz), as.double(sill), as.double(range), as.double(nugget), as.double(t(coordinates(samples))), as.double(srcData), as.integer(numSrc), as.integer(.covID(covmodel)), as.double(anis), as.integer(check), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), as.integer(gpu.cache), retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditionalSimUncondResiduals_3d", out=double((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditionalSimKrigeResiduals_3d", out=double(nx*ny*nz*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditionalSimUncondResiduals_3d", out=double(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditionalSimSimpleKrigeResiduals_3d", out=double(nx*ny*nz*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Simple Kriging Residuals"
				times = c(times,t1)
			}
		}		
		
		# clean up
		result = .C("conditionalSimRelease_3d",retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			

		out = as.vector(res$out)
		dim(out) = c(nx,ny,nz,k)		
		if (compute.stats) {
			mean_grid <- apply(out,1:(length(dim(out))-1), mean)
			sd_grid <- apply(out,1:(length(dim(out))-1), sd)
			out <- c(out,mean_grid,sd_grid)
			dim(out) = c(nx,ny,nz,k+2)
		}		
		if(as.sp) {
			if (compute.stats) {
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k+2,nrow = nx*ny*nz)))	
				names(out@data) = c(paste("sim",1:k,sep=""),"mean","sd")
			}
			else { 
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k,nrow = nx*ny*nz)))		
				names(out@data) = paste("sim",1:k,sep="")
			}
		}				
		if (!is.null(aggregation.features) && !is.null(aggregation.func)) {
			aggdata = over(aggregation.features,out,fn = aggregation.func)
			out = SpatialPolygonsDataFrame(aggregation.features,as.data.frame(aggdata))
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
		xmax = xmin + (nx-1) * dx
		ymax = ymin + (ny-1) * dy	
		zmax = zmin + (nz-1) * dz	
		
		
		retcode = 0
		
		if (benchmark) .gpuSimStartTimer()
		result = .C("unconditionalSimInit_3d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(zmin), as.double(zmax), as.integer(nz), as.double(sill), as.double(range), as.double(nugget), as.integer(.covID(covmodel)), as.double(anis), as.integer(check), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))			
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Initialization"
			times = c(times,t1)
		}
		
		if (benchmark) .gpuSimStartTimer()
		res = .C("unconditionalSimRealizations_3d", out=double(nx*ny*nz*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Generating Unconditional Realizations"
			times = c(times,t1)
		}
	
		result = .C("unconditionalSimRelease_3d", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			
		out = as.vector(res$out)
		dim(out) = c(nx,ny,nz,k)		
		if (compute.stats) {
			mean_grid <- apply(out,1:(length(dim(out))-1), mean)
			sd_grid <- apply(out,1:(length(dim(out))-1), sd)
			out <- c(out,mean_grid,sd_grid)
			dim(out) = c(nx,ny,nz,k+2)
		}		
		if(as.sp) {
			if (compute.stats) {
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k+2,nrow = nx*ny*nz)))	
				names(out@data) = c(paste("sim",1:k,sep=""),"mean","sd")
			}
			else { 
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k,nrow = nx*ny*nz)))		
				names(out@data) = paste("sim",1:k,sep="")
			}
		}
		if (!is.null(aggregation.features) && !is.null(aggregation.func)) {
			aggdata = over(aggregation.features,out,fn = aggregation.func)
			out = SpatialPolygonsDataFrame(aggregation.features,as.data.frame(aggdata))
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



 
 
 


