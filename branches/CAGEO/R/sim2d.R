 
 
 
 
.sim2d <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method = 'O', mu = 0, aggregation.features=NULL, aggregation.func=mean, gpu.cache = FALSE, as.sp = FALSE, neg.eigenvals.action = "ignore",eigenvals.tol=-1e-07, benchmark = FALSE, compute.stats = FALSE, anis=c(0,0,0,1,1), cpu.invertonly = FALSE, sim.n, sim.m) {
	
	if (benchmark) {
		times = c() # runtimes of single computation steps
		.gpuSimStartTimer()
	}
	out = 0
	
	if (!missing(uncond) && !missing(samples)) {
		#only conditioning, k is ignored and derived from uncond object
		cat("Performing two-dimensional conditioning in double precision...")
		cat("\n")
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
		xmax = xmin + (nx-1) * dx
		ymax = ymin + (ny-1) * dy	
		

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
		
		
		
		result = .C("conditioningInit_2d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.double(t(coordinates(samples))), as.double(srcData), as.integer(numSrc),  as.integer(k), as.double(uncond), as.integer(.covID(covmodel)), as.double(anis[1]), as.double(anis[4]), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), retcode = as.integer(retcode),PACKAGE="gpusim")
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
			res = .C("conditioningResiduals_2d", out=double((numSrc + 1) * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
			
			if (benchmark) .gpuSimStartTimer()
			
			# solve residual equation system
			dim(res$out) = c(numSrc+1,k)
			y <- 0
			if (cpu.invertonly) {
				y = solve(cov.l)
			}
			else {
				y = solve(cov.l, res$out)	
			}				
			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations
			if (benchmark) .gpuSimStartTimer()			
			res = .C("conditioningKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditioningResiduals_2d", out=double(numSrc * k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
					
			# solve residual equation system
			if (benchmark) .gpuSimStartTimer()	
			dim(res$out) = c(numSrc,k)
			y <- 0
			if (cpu.invertonly) {
				# y = solve(cov.l)
        message("Cholesky decomposition of simple kriging system.")
        cholesky=chol(cov.l)
        y <- chol2inv(cholesky)
			}
			else {
				y = solve(cov.l, res$out)	
			}				
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations	
			if (benchmark) .gpuSimStartTimer()	
			res = .C("conditioningSimpleKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Simple Kriging Residuals"
				times = c(times,t1)
			}
		}		
		
		result = .C("conditioningRelease_2d", retcode = as.integer(retcode),PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for simulation returned error:" , .gpuSimCatchError(result$retcode)))
			
		out = as.vector(res$out)
		dim(out) = c(nx,ny,k)		
		if (compute.stats) {
			mean_grid <- apply(out,1:(length(dim(out))-1), mean)
			sd_grid <- apply(out,1:(length(dim(out))-1), sd)
			out <- c(out,mean_grid,sd_grid)
			dim(out) = c(nx,ny,k+2)
		}		
		if(as.sp) {
			if (compute.stats) {
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k+2,nrow = nx*ny)))	
				names(out@data) = c(paste("sim",1:k,sep=""),"mean","sd")
			}
			else { 
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k,nrow = nx*ny)))		
				names(out@data) = paste("sim",1:k,sep="")
			}
		}	
		if (!is.null(aggregation.features) && !is.null(aggregation.func)) {
			aggdata = over(aggregation.features,out,fn = aggregation.func)
			out = SpatialPolygonsDataFrame(aggregation.features,as.data.frame(aggdata))
		}			
	}
	else if (!missing(k) && !missing(samples)) {
		#conditional simulation
		cat("Performing two-dimensional conditional simulation in double precision...")
		cat("\n")
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
		xmax = xmin + (nx-1) * dx 
		ymax = ymin + (ny-1) * dy	
		if (missing(sim.n)) sim.n=2*nx
		if (missing(sim.m)) sim.m=2*ny
		
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
		check = (neg.eigenvals.action == "error")
		seteigenvalszero = (neg.eigenvals.action == "setzero")
		
		if (benchmark) .gpuSimStartTimer()
		result = .C("conditionalSimInit_2d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.double(t(coordinates(samples))), as.double(srcData), as.integer(numSrc), as.integer(.covID(covmodel)), as.double(anis[1]), as.double(anis[4]), as.integer(check), as.integer(seteigenvalszero), as.double(eigenvals.tol), as.integer(.gpuSimKrigeMethod(kriging.method)), as.double(mu), as.integer(gpu.cache), as.integer(cpu.invertonly), as.integer(sim.n), as.integer(sim.m), retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditionalSimUncondResiduals_2d", out=double((numSrc + 1) * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
			
			# solve residual equation system
			if (benchmark) .gpuSimStartTimer()	
			dim(res$out) = c(numSrc+1,k)
			y <- 0
			if (cpu.invertonly) {
				y = solve(cov.l)
			}
			else {
				y = solve(cov.l, res$out)	
			}			
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations		
			if (benchmark) .gpuSimStartTimer()	
			res = .C("conditionalSimKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
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
			res = .C("conditionalSimUncondResiduals_2d", out=double(numSrc * k), as.integer(k),retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Computation of residuals between generated unconditional realizations and given data returned error: ",.gpuSimCatchError(res$retcode)))		
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Generation of Unconditional Realizations"
				times = c(times,t1)
			}
			
			# solve residual equation system
			if (benchmark) .gpuSimStartTimer()
			dim(res$out) = c(numSrc,k)
			y <- 0
			if (cpu.invertonly) {
			  # y = solve(cov.l)
			  message("Cholesky decomposition of simple kriging system.")
			  cholesky=chol(cov.l)
			  y <- chol2inv(cholesky)
			}
			else {
				y = solve(cov.l, res$out)	
			}				
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "CPU Solving Residual Equation System"
				times = c(times,t1)
			}
			
			# interpolate residuals and add to the unconditional realizations		
			if (benchmark) .gpuSimStartTimer()
			res = .C("conditionalSimSimpleKrigeResiduals_2d", out=double(nx*ny*k), as.double(y), retcode = as.integer(retcode), PACKAGE="gpusim")
			if (res$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ", .gpuSimCatchError(res$retcode)))
			if (benchmark) {
				t1 = .gpuSimStopTimer()
				names(t1) = "GPU Simple Kriging Residuals"
				times = c(times,t1)
			}
		}		
		
		# clean up
		result = .C("conditionalSimRelease_2d",retcode = as.integer(retcode),PACKAGE="gpusim")	
		if (result$retcode != 0) stop(paste("Releasing memory for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			

		out = as.vector(res$out)
		dim(out) = c(nx,ny,k)		
		if (compute.stats) {
			mean_grid <- apply(out,1:(length(dim(out))-1), mean)
			sd_grid <- apply(out,1:(length(dim(out))-1), sd)
			out <- c(out,mean_grid,sd_grid)
			dim(out) = c(nx,ny,k+2)
		}		
		if(as.sp) {
			if (compute.stats) {
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k+2,nrow = nx*ny)))	
				names(out@data) = c(paste("sim",1:k,sep=""),"mean","sd")
			}
			else { 
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k,nrow = nx*ny)))		
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
		cat("Performing two-dimensional unconditional simulation in double precision...")
		cat("\n")
		xmin = grid@cellcentre.offset[1]
		ymin = grid@cellcentre.offset[2]
		dx = grid@cellsize[1]
		dy = grid@cellsize[2]
		nx = grid@cells.dim[1]
		ny = grid@cells.dim[2]
		xmax = xmin + (nx-1) * dx
		ymax = ymin + (ny-1) * dy	
		if (missing(sim.n)) sim.n=2*nx
		if (missing(sim.m)) sim.m=2*ny
		
		retcode = 0
		check = (neg.eigenvals.action == "error")
		seteigenvalszero = (neg.eigenvals.action == "setzero")
		
		
		if (benchmark) .gpuSimStartTimer()
		result = .C("unconditionalSimInit_2d", as.double(xmin), as.double(xmax), as.integer(nx), as.double(ymin),as.double(ymax), as.integer(ny), as.double(sill), as.double(range), as.double(nugget), as.integer(.covID(covmodel)), as.double(anis[1]), as.double(anis[4]), as.integer(check), as.integer(seteigenvalszero),as.double(eigenvals.tol), as.integer(sim.n), as.integer(sim.m), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Initialization of unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))			
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Initialization"
			times = c(times,t1)
		}
		
		if (benchmark) .gpuSimStartTimer()
		res = .C("unconditionalSimRealizations_2d", out=double(nx*ny*k), as.integer(k), retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Generation of realizations for conditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
		if (benchmark) {
			t1 = .gpuSimStopTimer()
			names(t1) = "GPU Generating Unconditional Realizations"
			times = c(times,t1)
		}
	
		result = .C("unconditionalSimRelease_2d", retcode = as.integer(retcode), PACKAGE="gpusim")
		if (result$retcode != 0) stop(paste("Releasing memory for unconditional simulation returned error: ",.gpuSimCatchError(result$retcode)))
			
		out = as.vector(res$out)
		dim(out) = c(nx,ny,k)		
		if (compute.stats) {
			mean_grid <- apply(out,1:(length(dim(out))-1), mean)
			sd_grid <- apply(out,1:(length(dim(out))-1), sd)
			out <- c(out,mean_grid,sd_grid)
			dim(out) = c(nx,ny,k+2)
		}		
		if(as.sp) {
			if (compute.stats) {
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k+2,nrow = nx*ny)))	
				names(out@data) = c(paste("sim",1:k,sep=""),"mean","sd")
			}
			else { 
				out = SpatialGridDataFrame(grid,as.data.frame(matrix(out,ncol = k,nrow = nx*ny)))		
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





