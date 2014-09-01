Conugate Gradient for Least Squares in CUDA
===========================================

This is a CUDA implementation of [CGLS](http://web.stanford.edu/group/SOL/software/cgls/) for sparse matrices. CGLS solves problem

```
minimize ||Ax - b||_2^2 + s ||x||_2^2,
```

by using the [Conjugate Gradient method](http://en.wikipedia.org/wiki/Conjugate_gradient_method) (CG). It is more numerically stable than simply applying CG to the normal equations. The implementation supports both CSR and CSC matrices in single and double precision. 


Requirements
============
You will need a CUDA capable GPU along with the CUDA SDK installed on your computer. We use the  cuSPARSE and cuBLAS libraries.

Instructions
============
Clone the repository and type `make test`. It should work out of the box if you're on 64-bit Linux system and installed the CUDA SDK to its default location (/usr/local/cuda). If not, you may need to modify the `Makefile`.  
