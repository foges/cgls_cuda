Conjugate Gradient for Least Squares in CUDA
===========================================

This is a CUDA implementation of [CGLS](http://web.stanford.edu/group/SOL/software/cgls) for sparse matrices. CGLS solves problem

```
minimize ||Ax - b||_2^2 + s ||x||_2^2,
```

by using the [Conjugate Gradient method](http://en.wikipedia.org/wiki/Conjugate_gradient_method) (CG). It is more numerically stable than simply applying CG to the normal equations. The implementation supports any combination of real and complex valued matrices, CSR and CSC format, and single and double precision. Additionally abstract operators for computing `Ax` and `A^Tx` may be used instead of sparse matrices.

####Performance

CGLS was run on two of the largest non-square matrices in [Tim Davis' sparse matrix collection](http://www.cise.ufl.edu/research/sparse/matrices) on an Nvidia Tesla K40c. 

| Matrix name              |  Dimension       | Non-zeros      | Iter. | Time  | Time / (iter * nnz) |
|--------------------------|:----------------:|----------------|-------|-------|---------------------|
| Yoshiyasu.mesh_grid      | (230k, 9k)       | 850k           | 794   | 0.52 s| 0.77 ns             |
| JGD_GL7d.GL7d18          | (2M, 1.5M)       | 36M            | 77    | 3.7 s | 1.3 ns              |
| ~U[-1, 1]                | (1M, 950k)       | 250M           | 342   | 95 s  | 1.1 ns              |

In each instance there was no shift (i.e. `s = 0`), the tolerance was set to `1e-6`, and the arithmetic was performed in double precision.


####Example Usage - CSR Matrix

To solve a least squares problem where the matrix is real-valued, and stored in double precision CSR format, use the syntax

```
cgls::Solve<double, cgls::CSR>(val, rptr, cind, m, n, nnz, b, x, s, tol, maxit, quiet)
```
The arguments are (note that all arrays must be in GPU memory):

  + `(double*) val`: Array of matrix entries. The length should be `nnz`.
  + `(int*) rptr`: Array of row pointers. The length should be `m+1`.
  + `(int*) cind`: Array of column indicies. The length should be `nnz`.
  + `(int) m`: Number of rows in `A`.
  + `(int) n`: Number of columns in `A`.
  + `(int) nnz`: Number of non-zero entries in `A`.
  + `(double*) b`: Left-hand-side array. The length should be `m`.
  + `(double*) x`: Pointer to where the solution should be stored and at the same time it serves as an initial guess. It is very important to initialize `x` (eg. to 0) before calling the solver.
  + `(double) s`: Shift term (the same as `s` in the objective function).
  + `(double) tol`: Relative tolerance to which the problem should be solved (recommended 1e-6).
  + `(int) maxit`: Maximum number of iterations before the solver stops (recommended 20-100, but it depends heavily on the condition number of `A`).
  + `(bool) quiet`: Disables output to the console if set to `true`.
  
####Example Usage - Abstract Operator

You may also want to use CGLS if you have an abstract operator that computes `Ax` and `A^Tx`. To do so, define a GEMV-like functor that inherits from the abstract class

```
template <typename T>
struct Gemv {
  virtual ~Gemv() { };
  virtual int operator()(char op, const T alpha, const T *x, const T beta, T *y) = 0;
};
```
When invoked, the functor should compute `y := alpha*op(A)x + beta*y`, where `op` is either `'n'` or `'t'`, corresponding to `Ax` and `A^Tx` (or `A^Hx` in the complex case). The functor should return a non-zero value if unsuccessful and 0 if the operation succeeded. Once the functor is defined, you can invoke `CGLS` with 

```
cgls::Solve(cublas_handle, A, m, n, b, x, shift, tol, maxit, quiet);
```

The arguments are:

  + `(cublasHandle_t) cublas_handle`: An initialized cuBLAS handle.
  + `(const cgls::Gemv<double>&) A`: An instance of the abstract operator.
  + ... (the rest of the arguments are the same as above).
  
####Return values

Upon exit, CGLS will have modified the `x` argument and return an `int` flag corresponding to one of the error codes

    0 : CGLS converged to the desired tolerance tol within maxit iterations.
    1 : The vector b had norm less than eps, solution likely x = 0.
    2 : CGLS iterated maxit times but did not converge.
    3 : Matrix (A'*A + shift*I) seems to be singular or indefinite.
    4 : Likely instable, (A'*A + shift*I) indefinite and norm(x) decreased.
    5 : Error in applying operator A.
    6 : Error in applying operator A^T.

####Requirements

You will need a CUDA capable GPU along with the CUDA SDK installed on your computer. We use the  cuSPARSE and cuBLAS libraries, which are part of the CUDA SDK.

####Instructions

Clone the repository and type `make test`. It should work out of the box if you're on 64-bit Linux system and installed the CUDA SDK to its default location (/usr/local/cuda). If not, you may need to modify the `Makefile`.

####Acknowledgement

The code is based on an implementation by Michael Saunders. Matlab code can be found on the [Stanford Systems Optimization Lab - CGLS website](http://web.stanford.edu/group/SOL/software/cgls/).
