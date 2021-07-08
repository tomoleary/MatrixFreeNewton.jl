			Prototyping of matrix free Newton methods in Julia

	      ___           ___                       ___                       ___       
	     /__/\         /  /\          ___        /  /\        ___          /__/|      
	    |  |::\       /  /::\        /  /\      /  /::\      /  /\        |  |:|      
	    |  |:|:\     /  /:/\:\      /  /:/     /  /:/\:\    /  /:/        |  |:|      
	  __|__|:|\:\   /  /:/~/::\    /  /:/     /  /:/~/:/   /__/::\      __|__|:|      
	 /__/::::| \:\ /__/:/ /:/\:\  /  /::\    /__/:/ /:/___ \__\/\:\__  /__/::::\____  
	 \  \:\~~\__\/ \  \:\/:/__\/ /__/:/\:\   \  \:\/:::::/    \  \:\/\    ~\~~\::::/  
	  \  \:\        \  \::/      \__\/  \:\   \  \::/~~~~      \__\::/     |~~|:|~~   
	   \  \:\        \  \:\           \  \:\   \  \:\          /__/:/      |  |:|     
	    \  \:\        \  \:\           \__\/    \  \:\         \__\/       |  |:|     
	     \__\/         \__\/                     \__\/                     |__|/      
	               ___         ___           ___           ___                        
	              /  /\       /  /\         /  /\         /  /\                       
	             /  /:/_     /  /::\       /  /:/_       /  /:/_                      
	            /  /:/ /\   /  /:/\:\     /  /:/ /\     /  /:/ /\                     
	           /  /:/ /:/  /  /:/~/:/    /  /:/ /:/_   /  /:/ /:/_                    
	          /__/:/ /:/  /__/:/ /:/___ /__/:/ /:/ /\ /__/:/ /:/ /\                   
	          \  \:\/:/   \  \:\/:::::/ \  \:\/:/ /:/ \  \:\/:/ /:/                   
	           \  \::/     \  \::/~~~~   \  \::/ /:/   \  \::/ /:/                    
	            \  \:\      \  \:\        \  \:\/:/     \  \:\/:/                     
	             \  \:\      \  \:\        \  \::/       \  \::/                      
	              \__\/       \__\/         \__\/         \__\/                       
	      ___           ___           ___                       ___           ___     
	     /__/\         /  /\         /__/\          ___        /  /\         /__/\    
	     \  \:\       /  /:/_       _\_ \:\        /  /\      /  /::\        \  \:\   
	      \  \:\     /  /:/ /\     /__/\ \:\      /  /:/     /  /:/\:\        \  \:\  
	  _____\__\:\   /  /:/ /:/_   _\_ \:\ \:\    /  /:/     /  /:/  \:\   _____\__\:\ 
	 /__/::::::::\ /__/:/ /:/ /\ /__/\ \:\ \:\  /  /::\    /__/:/ \__\:\ /__/::::::::\
	 \  \:\~~\~~\/ \  \:\/:/ /:/ \  \:\ \:\/:/ /__/:/\:\   \  \:\ /  /:/ \  \:\~~\~~\/
	  \  \:\  ~~~   \  \::/ /:/   \  \:\ \::/  \__\/  \:\   \  \:\  /:/   \  \:\  ~~~ 
	   \  \:\        \  \:\/:/     \  \:\/:/        \  \:\   \  \:\/:/     \  \:\     
	    \  \:\        \  \::/       \  \::/          \__\/    \  \::/       \  \:\    
	     \__\/         \__\/         \__\/                     \__\/         \__\/    

[![DOI](https://zenodo.org/badge/378233621.svg)](https://zenodo.org/badge/latestdoi/378233621)
[![License](https://img.shields.io/github/license/tomoleary/MatrixFreeNewton.jl)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/tomoleary/MatrixFreeNewton.jl)](https://julialang.org/)
![Code size](https://img.shields.io/github/languages/code-size/tomoleary/MatrixFreeNewton.jl)
[![Issues](https://img.shields.io/github/issues/hippylib/hippyflow)](https://github.com/tomoleary/MatrixFreeNewton.jl/issues)
[![Latest commit](https://img.shields.io/github/last-commit/hippylib/hippyflow)](https://github.com/tomoleary/MatrixFreeNewton.jl/commits/master)


The Hessian action is exposed via matrix-vector products:
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?H\widehat{w}=\frac{d}{dw}(g^T\widehat{w})" /> 
</p>

and matrix-matrix products:
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?H\widehat{W}=\frac{d}{dw}(g^T\widehat{W})" /> 
</p>




# References

These publications motivate and use the hessianlearn library for stochastic nonconvex optimization

- \[1\]  O'Leary-Roseberry, T., Alger, N., Ghattas O.,
[**Low Rank Saddle Free Newton**](https://arxiv.org/abs/2002.02881).
arXiv:2002.02881.
([Download](https://arxiv.org/pdf/2002.02881.pdf))<details><summary>BibTeX</summary><pre>
@article{o2020low,
  title={Low Rank Saddle Free Newton: Algorithm and Analysis},
  author={O'Leary-Roseberry, Thomas and Alger, Nick and Ghattas, Omar},
  journal={arXiv preprint arXiv:2002.02881},
  year={2020}
}
}</pre></details>


