

gpu.Exp.image.cov <- function(ind1, ind2, Y, cov.obj = NULL, setup = FALSE, grid, ...) 
{
    require(fields)
    require(gpusim)
    if (is.null(cov.obj)) {
        dx <- grid$x[2] - grid$x[1]
        dy <- grid$y[2] - grid$y[1]
        m <- length(grid$x)
        n <- length(grid$y)
        M <- ceiling2(2 * m)
        N <- ceiling2(2 * n)
        xg <- make.surface.grid(list((1:M) * dx, (1:N) * dy))
        center <- matrix(c((dx * M)/2, (dy * N)/2), nrow = 1, 
            ncol = 2)
        out <- Exp.cov(xg, center, ...) ## FORTRAN CALL!!!!
        out <- as.surface(xg, c(out))$z
        temp <- matrix(0, nrow = M, ncol = N)
        temp[M/2, N/2] <- 1
        wght <- gpuFFT(out)/(gpuFFT(temp) * M * N)
        cov.obj <- list(m = m, n = n, grid = grid, N = N, M = M, 
            wght = wght, call = match.call())
        if (setup) {
            return(cov.obj)
        }
    }
    temp <- matrix(0, nrow = cov.obj$M, ncol = cov.obj$N)
    if (missing(ind1)) {
        temp[1:cov.obj$m, 1:cov.obj$n] <- Y
        Re(gpuFFT(gpuFFT(temp) * cov.obj$wght, inv = TRUE)[1:cov.obj$m, 
            1:cov.obj$n])
    }
    else {
        if (missing(ind2)) {
            temp[ind1] <- Y
        }
        else {
            temp[ind2] <- Y
        }
        Re(gpuFFT(gpuFFT(temp) * cov.obj$wght, inv = TRUE)[ind1])
    }
}




gpu.sim.rf <- function(obj) 
{
    n <- obj$n
    m <- obj$m
    M <- obj$M
    N <- obj$N
    if (any(Re(obj$wght) < 0)) {
        stop("FFT of covariance has negative\nvalues")
    }
    z <- gpuFFT(matrix(rnorm(N * M), ncol = N, nrow = M))
    Re(gpuFFT(sqrt(obj$wght) * z, inv = TRUE))[1:m, 1:n]/sqrt(M * N)
}




gpu.sim.Krig.grid <- function (object, grid.list = NA, M = 1, nx = 40, ny = 40, xy = c(1,2), verbose = FALSE, sigma2 = NA, rho = NA, extrap = FALSE) 
{
    if (object$cov.function.name != "stationary.cov") {
        stop("covariance function is not stationary.cov")
    }
    if (is.na(grid.list)[1]) {
        if (is.null(object$x)) {
            stop("Need a an X matrix in the output object")
        }
        grid.list <- fields.x.to.grid(object$x, nx = nx, ny = ny, 
            xy = xy)
    }
    temp <- parse.grid.list(grid.list)
    nx <- temp$nx
    ny <- temp$ny
    glist <- list(x = temp$x, y = temp$y)
    if (is.na(sigma2)) {
        sigma2 <- object$best.model[2]
    }
    if (is.na(rho)) {
        rho <- object$best.model[3]
    }
    m <- nx * ny
    n <- nrow(object$xM)
    N <- n
    if (verbose) {
        cat(" m,n,N, sigma2, rho", m, n, N, sigma2, rho, fill = TRUE)
    }
    xc <- object$transform$x.center
    xs <- object$transform$x.scale
    if (verbose) {
        cat("center and scale", fill = TRUE)
        print(xc)
        print(xs)
    }
    cov.obj <- do.call("gpu.stationary.image.cov", c(object$args, 
        list(setup = TRUE, grid = glist)))
    out <- array(NA, c(nx, ny, M))
    h.hat <- predict.surface(object, grid.list = grid.list, extrap = extrap)$z
    if (verbose) {
        cat("mean predicted field", fill = TRUE)
        image.plot(h.hat)
    }
    h.true <- list(x = glist$x, y = glist$y, z = matrix(NA, nx, 
        ny))
    W2i <- Krig.make.Wi(object, verbose = verbose)$W2i
    if (verbose) {
        cat("dim of W2i", dim(W2i), fill = TRUE)
    }
    for (k in 1:M) {
        h.true$z <- sqrt(object$rhohat) * gpu.sim.rf(cov.obj)
        if (verbose) {
            cat("mean predicted field", fill = TRUE)
            image.plot(h.true)
        }
        h.data <- interp.surface(h.true, object$xM)
        if (verbose) {
            cat("synthetic true values", h.data, fill = TRUE)
        }
        y.synthetic <- h.data + sqrt(sigma2) * W2i %d*% rnorm(N)
        if (verbose) {
            cat("synthetic data", y.synthetic, fill = TRUE)
        }
        temp.error <- predict.surface(object, grid.list = grid.list, 
            yM = y.synthetic, eval.correlation.model = FALSE, 
            extrap = TRUE)$z - h.true$z
        if (verbose) {
            cat("mean predicted field", fill = TRUE)
            image.plot(temp.error)
        }
        out[, , k] <- h.hat + temp.error
    }
    return(list(x = glist$x, y = glist$y, z = out))
}







gpu.stationary.image.cov <- function (ind1, ind2, Y, cov.obj = NULL, setup = FALSE, grid, 
    M = NULL, N = NULL, Covariance = "Matern", Distance = "rdist", 
    ...) 
{
    if (is.null(cov.obj)) {
        dx <- grid$x[2] - grid$x[1]
        dy <- grid$y[2] - grid$y[1]
        m <- length(grid$x)
        n <- length(grid$y)
        if (is.null(M)) 
            M <- (2 * m)
        if (is.null(N)) 
            N <- (2 * n)
        xg <- make.surface.grid(list((1:M) * dx, (1:N) * dy))
        center <- matrix(c((dx * M)/2, (dy * N)/2), nrow = 1, 
            ncol = 2)
        out <- stationary.cov(xg, center, Covariance = Covariance, 
            Distance = Distance, ...)
        out <- matrix(c(out), nrow = M, ncol = N)
        temp <- matrix(0, nrow = M, ncol = N)
        temp[M/2, N/2] <- 1
        wght <- gpuFFT(out)/(gpuFFT(temp) * M * N)
        cov.obj <- list(m = m, n = n, grid = grid, N = N, M = M, 
            wght = wght, call = match.call())
        if (setup) {
            return(cov.obj)
        }
    }
    temp <- matrix(0, nrow = cov.obj$M, ncol = cov.obj$N)
    if (missing(ind1)) {
        temp[1:cov.obj$m, 1:cov.obj$n] <- Y
        Re(gpuFFT(gpuFFT(temp) * cov.obj$wght, inv = TRUE)[1:cov.obj$m, 
            1:cov.obj$n])
    }
    else {
        if (missing(ind2)) {
            temp[ind1] <- Y
        }
        else {
            temp[ind2] <- Y
        }
        Re(gpuFFT(gpuFFT(temp) * cov.obj$wght, inv = TRUE)[ind1])
    }
}






