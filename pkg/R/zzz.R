.packageName <- "gpusim" 

.First.lib  <-  function(libname, pkgname)  {
 	library.dynam("gpusim", pkgname, libname)
 
	## Initialize cuda device
	init = .C("init", res = integer(1), PACKAGE="gpusim")
	if (init$res != 0) {
		stop("Error: Initialization of CUDA device failed! Do you have a capable gpu and a properly installed CUDA runtime environment?")
	}
}


.Last.lib  <- function(libpath)  {
 library.dynam.unload("gpusim", libpath)
}