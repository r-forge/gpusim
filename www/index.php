
<!-- This is the project specific website template -->
<!-- It can be changed as liked or replaced by other content -->

<?php

$domain=ereg_replace('[^\.]*\.(.*)$','\1',$_SERVER['HTTP_HOST']);
$group_name=ereg_replace('([^\.]*)\..*$','\1',$_SERVER['HTTP_HOST']);
$themeroot='r-forge.r-project.org/themes/rforge/';

echo '<?xml version="1.0" encoding="UTF-8"?>';
?>
<!DOCTYPE html
	PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
	"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en   ">

  <head>
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
	<title><?php echo $group_name; ?></title>
	<link href="http://<?php echo $themeroot; ?>styles/estilo1.css" rel="stylesheet" type="text/css" />
  </head>

<body style="margin:auto;width:800px;text-align:justify;font-family:sans-serif;">


<!-- get project title  -->
<!-- own website starts here, the following may be changed as you like -->

<div style="text-align:center">
<a href="http://r-forge.r-project.org/"><img src="http://<?php echo $themeroot; ?>/imagesrf/logo.png" border="0" alt="R-Forge Logo" /> </a>
</div>

<a name="Top"></a>



<div style="text-align:center">
<a href="#introduction">Introduction</a>
<a href="#requirements">System Requirements</a>
<a href="#download">Download</a>
<a href="#start">Getting Started</a>
<a href="#example">Example Results</a>
<a href="#contact">Contact</a>
<a href="#references">References</a>
</div>


<div style="text-align:center">
<h1>The gpusim R package</h1>

</div>


<h3><a name="introduction">Introduction</a></h3>
The open-source gpusim R package provides fast functions for the simulation of gaussian random fields using graphics processing units.
Based on NVIDIA's CUDA framework our packages makes use of the cufft and curand libraries. Both, the generation of unconditional simulations as well as the following conditioning step is implemented for benefit from current GPU's architecture.
However, the performance of the simulation strongly depends on the avaiable hardware, the best speedup factors are achieved for large grids. You can find some performance analysis in [1]. 
The gpusim R package is built according to the well known packages <i>fields</i> [2] and <i>gstat</i> [3] and enables converting simulation results to <i>sp</i> [4] objects.
We would be happy if you let us know about your experiences while testing the package.


<br/><br/>

<table style="border:1px solid gray;width:750px;margin:auto;">
<tr>
<td><img src="images/uncondExp1.png" width="150"/></td>
<td><img src="images/uncondExp2.png" width="150"/></td>
<td><img src="images/uncondExp3.png" width="150"/></td>
<td><img src="images/uncondGau2.png" width="150"/></td>
<td><img src="images/uncondExpNugget15.png" width="150"/></td>
</table>
</div>


<h3><a name="requirements">System Requirements</a></h3>
Using gpusim requires the latest R version and a appropriate GPU. You will also need latest graphics drivers including NVIDIA's CUDA libraries.
For compiling from source, you need NVIDIA's CUDA SDK. A detailed  (but short) installation instruction can be found in the package <a href="">vignette</a>.
 



<h3><a name="download">Download</a></h3>
The current version is 0.01. Downloads for different platforms can be found below. Note that there is no x64 windows binary available at the moment but planned for future releases.
For compiling from sources on Windows, you need a Microsoft Visual Studio 2010 Compiler and NVIDIA's GPU Computing SDK.
<br/><br/>
<a href="javascript:alert('Not available at the moment!')">Source - tar.gz</a><br/>
<a href="javascript:alert('Not available at the moment!')">Windows Binaries (x86) - .zip</a><br/>
<a href="javascript:alert('Not available at the moment!')">Windows Binaries (x64) - .zip</a><br/>


<h3><a name="start">Getting Started</a></h3>

As a starting point for working with gpusim, watch the following R script which generates 5 realizations of an unconditional simulation using an exponential covariance function.
Before you run the following script, make sure you have successfully installed <b>gpusim</b> by executing <span style="font-family:Monospace;">R CMD INSTALL gpusim</span> in your Linux terminal or installing the package binaries using the Rgui in Windows.
<div style="font-family:Monospace;background-color:#EEEEEE;margin-top:5px;margin-bottom:5px;border:1px solid black;">
library(gpusim) # <-- load R package<br/>
gpuDeviceInfo() # <-- print GPU capabilities<br/>
<br/>
<br/>
# define grid<br/>
xmin = 0<br/>
xmax = 5<br/>
ymin = 0<br/>
ymax = 5<br/>
nx = 100<br/>
ny = 100<br/>
dx = (xmax-xmin)/nx<br/>
dy = (ymax-ymin)/ny<br/>
grid = GridTopology(c(xmin,ymin), c(dx,dy), c(nx,ny))<br/>
<br/>
# define covariance function<br/>
model = "Exp"<br/>
range = 0.5<br/>
sill = 3<br/>
nugget = 0<br/>
<br/>
k = 5  # number of realizations<br/>
<br/>
simGPU = gpuSim(grid, model, sill, range, nugget, k) # <-- run simulation<br/>
image.plot(simGPU[,,5]) # <-- plot 5-th realization<br/>
</div>





<h3><a name="example">Example Results</a></h3>


<table style="border:1px solid gray;width:750px;margin:auto;">
<tr>
<td><img src="images/uncondExp1.png" width="250"/></td>
<td><img src="images/uncondExp2.png" width="250"/></td>
<td><img src="images/uncondExp3.png" width="250"/></td>
</tr>
<tr>
<td colspan="3">
Three equiprobable realizations of an unconditional simulation on a 500x500 grid using an exponential covariance function.
</td>
</tr>
</table>
<br/>


<table style="border:1px solid gray;width:750px;margin:auto;">
<tr>
<td><img src="images/uncondGau1.png" width="250"/></td>
<td><img src="images/uncondGau2.png" width="250"/></td>
<td><img src="images/uncondGau3.png" width="250"/></td>
</tr>
<tr>
<td colspan="3">
Three equiprobable realizations of an unconditional simulation on a 500x500 grid using a gaussian covariance function.
</td>
</tr>
</table>
<br/>





<table style="border:1px solid gray;width:750px;margin:auto;">
<tr>
<td><img src="images/uncondExpNugget0.png" width="250"/></td>
<td><img src="images/uncondExpNugget04.png" width="250"/></td>
<td><img src="images/uncondExpNugget15.png" width="250"/></td>
</tr>
<tr>
<td colspan="3">
Three unconditional simulation results with the same exponential covariance function but increasing nugget effect
</td>
</tr>
</table>
<br/>



<table style="border:1px solid gray;width:750px;margin:auto;">
<tr>
<td><img src="images/condExpPoints.png" width="250"/></td>
<td><img src="images/condExp.png" width="250"/></td>
<td><img src="images/varioresult.png" width="250"/></td>
</tr>
<tr>
<td colspan="3">
Conditional simulation on a 500x500 grid given 50 random points. The right figure compares the averaged resulting experimental variogram (points) of all realizations against the theoretical variogram model.
</td>
</tr>
</table>
<br/>



<h3><a name="contact">Contact</a></h3>
Katharina Henneboehl - katharina.henneboehl@uni-muenster.de<br/>
Marius Appel - marius.appel@uni-muenster.de



<h3><a name="references">References</a></h3>

<p>
[1] -
</p>


<p>
[2] Reinhard Furrer, Douglas Nychka and Stephen Sain (2011). fields:
  Tools for spatial data. R package version 6.6.1.
  <a href="http://CRAN.R-project.org/package=fields">http://CRAN.R-project.org/package=fields</a>
</p>

<p>
[3] Pebesma, E.J., 2004. Multivariable geostatistics in S: the gstat
  package. Computers & Geosciences, 30: 683-691.
</p>

<p>
[4] Pebesma, E.J., R.S. Bivand, 2005. Classes and methods for spatial
  data in R. R News 5 (2), <a href="http://cran.r-project.org/doc/Rnews/">http://cran.r-project.org/doc/Rnews/</a>.
</p>

<div style="text-align:center"><a href="#Top">Top</a></div>









<!-- end of project description -->

</body>
</html>
