.packageName <- "gpusim" 


# Error codes must be in accordance to c code
 .gpuSimCatchError <- function(code) {
	messages = c("successful", "fft of covariance matrix contains negative real parts", "unknown error returned", "no device found")
	return (messages[code+1])
 }
 
 # Kriging method codes must be in accordance to c code
 .gpuSimKrigeMethod <- function(identifier) {
	if (any(c('S','s') == identifier)) return(0) # simple
	if (any(c('O','o') == identifier)) return(1)# ordinary
	return (-1);
 }



.onLoad  <-  function(libname, pkgname)  {
 	library.dynam("gpusim", pkgname, libname)
 
	## Initialize cuda device
	init = .C("initSim", res = integer(1), PACKAGE="gpusim")
	if (init$res != 0) {
		stop(paste("Initialization of CUDA device failed:", .gpuSimCatchError()))
	}
	.C("initFFT",PACKAGE="gpusim")
}


.onUnload  <- function(libpath)  {
 library.dynam.unload("gpusim", libpath)
}