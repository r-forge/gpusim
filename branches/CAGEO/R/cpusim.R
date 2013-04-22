 



cpuSim <- function(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method='O', mu=0, aggregation.features=NULL, aggregation.func=mean, as.sp=FALSE, neg.eigenvals.action = "ignore", benchmark=FALSE, compute.stats=FALSE, anis=c(0,0,0,1,1)) {
	
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
		out <- .cpusim2d(grid, covmodel, sill, range, nugget, k, samples, uncond, kriging.method, mu, aggregation.features, aggregation.func, as.sp, neg.eigenvals.action, benchmark, compute.stats, anis)
	}
	else if (dims == 3) {
		stop("Three dimensional CPU simulation not supported")
	}
	else stop("Only two- or three-dimensional simulation supported!")
	
	
	
	return(out)
}








