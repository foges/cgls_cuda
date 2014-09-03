Conugate Gradient for Least Squares in CUDA
===========================================

This is a CUDA implementation of [CGLS](http://web.stanford.edu/group/SOL/software/cgls) for sparse matrices. CGLS solves problem

```
minimize ||Ax - b||_2^2 + s ||x||_2^2,
```

by using the [Conjugate Gradient method](http://en.wikipedia.org/wiki/Conjugate_gradient_method) (CG). It is more numerically stable than simply applying CG to the normal equations. The implementation supports both CSR and CSC matrices in single and double precision. 

Performance
-----------

CGLS was run on two of the largest non-square sparse matrices in [Tim Davis' sparse matrix collection](http://www.cise.ufl.edu/research/sparse/matrices) on an Nvidia Tesla K40c. 

| Matrix Name          |  Dimensions      | Non-Zeros      | Iterations | Time  |
|----------------------|:----------------:|----------------|------------|-------|
| JGD_GL7d / GL7d18    | (2e6, 1.5e6)     | 36e6           | 77         | 3.7 s |
| Yoshiyasu / mesh_grid| (2.3e5, 9e3)     | 8.5e5          | 794        | 0.52 s|

In each instance there was no shift (i.e. `s = 0`), the tolerance was set to `1e-6`, and the arithmetic was performed in double precision.


Example Usage
-------------
To solve a least squares problem where the matrix is stored in double precision CSR format, use the syntax

```
int flag = cgls::solve<double, cgls::CSR>(val, row_ptr, col_ind, m, n, nnz, b, x, shift, tol, maxit, quiet);
```
The arguments are (note that all arrays must be in GPU memory):
  + `(double*) val`: Array of matrix entries. The length should be `nnz`.
  + `(int*) row_ptr`: Array of row pointers. The length should be `m+1`.
  + `(int*) col_ind`: Array of column indicies. The length should be `nnz`.
  + `(int) m`: Number of rows in `A`.
  + `(int) n`: Number of columns in `A`.
  + `(int) nnz`: Number of non-zero entries in `A`.
  + `(double*) b`: Left-hand-side array. The length should be `m`.
  + `(double*) x`: Pointer to where the solution should be stored and at the same time it serves as an initial guess. It is very important to initialize `x` (eg. to 0) before calling the solver.
  + `(double) shift`: Shift term (the same as `s` in the objective function.
  + `(double) tol`: Relative tolerance to which the problem should be solved (recommended 1e-6).
  + `(int) maxit`: Maximum number of iterations before the solver stops (recommended 20-100, but it depends heavily on the condition number of `A`).
  + `(bool) quiet`: Disables output to the console if set to `true`.

Returns:
  + `(int) flag`: Status of the solver upon exiting (see `cgls.cuh` for an explanation of the error codes).

Requirements
------------
You will need a CUDA capable GPU along with the CUDA SDK installed on your computer. We use the  cuSPARSE and cuBLAS libraries.

Instructions
------------
Clone the repository and type `make test`. It should work out of the box if you're on 64-bit Linux system and installed the CUDA SDK to its default location (/usr/local/cuda). If not, you may need to modify the `Makefile`.

Acknowledgement
---------------
The code is based on an implementation by Michael Saunders.
