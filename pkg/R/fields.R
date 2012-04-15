

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


