//  CGLS Conjugate Gradient Least Squares
//  Attempts to solve the least squares problem
// 
//    min. ||A * x - b||_2
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  F          - Sparse ordering (CSC or CSR).
//
//  Function Arguments:
//  handle_s   - Cusparse handle.
// 
//  handle_b   - Cublas handle.
//
//  descr      - Cusparse matrix descriptor (i.e. 0- or 1-based indexing)
//
//  val        - Array of matrix values. The array should be of length nnz.
//
//  ptr        - Column pointer if (F is CSC) or row pointer if (F is CSR).
//               The array should be of length m+1.
//
//  ind        - Row indices if (F is CSC) or column indices if (F is CSR).
//               The array should be of length nnz.
//
//  (m, n)     - Matrix dimensions of A.
//
//  nnz        - Number of non-zeros in A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended >20).
//
//  quiet      - Disable printing to console.
//
//  Returns:
//  0 => CGLS converged to the desired tolerance tol within maxit iterations.
//  1 => The vector b had norm less than eps, solution likely x = 0.
//  2 => CGLS iterated maxit times but did not converge.
//  3 => Matrix (A'*A + shift*I) seems to be singular or indefinite.
//  4 => Likely instable, (A'*A + shift*I) indefinite and norm(x) decreased.
//
//  Reference:
//  http://web.stanford.edu/group/SOL/software/cgls/
//
#ifndef CGLS_CUH_
#define CGLS_CUH_

#include <stdio.h>

#include <algorithm>

#include <cublas_v2.h>
#include <cusparse.h>

namespace cgls {

// Data type for indices.
typedef int INT;

// Data type for sparse format.
enum CGLS_FMT { CSC, CSR };

// Sparse matrix-vector multiply templates.
template <typename T, CGLS_FMT F>
cusparseStatus_t spmv(cusparseHandle_t handle, cusparseOperation_t transA,
                      INT m, INT n, INT nnz, const T *alpha,
                      cusparseMatDescr_t descrA, const T *val, const INT *ptr,
                      const INT *ind, const T *x, const T *beta, T *y);

template <>
cusparseStatus_t spmv<double, CSR>(cusparseHandle_t handle,
                                   cusparseOperation_t transA, INT m, INT n,
                                   INT nnz, const double *alpha,
                                   cusparseMatDescr_t descrA, const double *val,
                                   const INT *ptr, const INT *ind,
                                   const double *x, const double *beta,
                                   double *y) {
  return cusparseDcsrmv(handle, transA, static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(nnz), alpha, descrA, val,
      static_cast<const int*>(ptr), static_cast<const int*>(ind), x, beta, y);
}

template <>
cusparseStatus_t spmv<double, CSC>(cusparseHandle_t handle,
                                   cusparseOperation_t transA, INT m, INT n,
                                   INT nnz, const double *alpha,
                                   cusparseMatDescr_t descrA, const double *val,
                                   const INT *ptr, const INT *ind,
                                   const double *x, const double *beta,
                                   double *y) {
  if (transA ==	CUSPARSE_OPERATION_TRANSPOSE)
    transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    transA = CUSPARSE_OPERATION_TRANSPOSE;
  return cusparseDcsrmv(handle, transA, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(nnz), alpha, descrA, val,
      static_cast<const int*>(ptr), static_cast<const int*>(ind), x, beta, y);
}

template <>
cusparseStatus_t spmv<float, CSR>(cusparseHandle_t handle,
                                  cusparseOperation_t transA, INT m, INT n,
                                  INT nnz, const float *alpha,
                                  cusparseMatDescr_t descrA, const float *val,
                                  const INT *ptr, const INT *ind,
                                  const float *x, const float *beta,
                                  float *y) {
  return cusparseScsrmv(handle, transA, static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(nnz), alpha, descrA, val,
      static_cast<const int*>(ptr), static_cast<const int*>(ind), x, beta, y);
}

template <>
cusparseStatus_t spmv<float, CSC>(cusparseHandle_t handle,
                                  cusparseOperation_t transA, INT m, INT n,
                                  INT nnz, const float *alpha,
                                  cusparseMatDescr_t descrA, const float *val,
                                  const INT *ptr, const INT *ind,
                                  const float *x, const float *beta,
                                  float *y) {
  if (transA ==	CUSPARSE_OPERATION_TRANSPOSE)
    transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  else
    transA = CUSPARSE_OPERATION_TRANSPOSE;
  return cusparseScsrmv(handle, transA, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(nnz), alpha, descrA, val,
      static_cast<const int*>(ptr), static_cast<const int*>(ind), x, beta, y);
}

// Templated 2-Norm blas function.
cublasStatus_t nrm2(cublasHandle_t handle, INT n, const double *x, INT incx,
                    double *result) {
  return cublasDnrm2(handle, static_cast<int>(n), x, 
      static_cast<int>(incx), result);
}

cublasStatus_t nrm2(cublasHandle_t handle, INT n, const float *x, INT incx,
                    float *result) {
  return cublasSnrm2(handle, static_cast<int>(n), x,
      static_cast<int>(incx), result);
}

// Templated axpy blas function.
cublasStatus_t axpy(cublasHandle_t handle, INT n, double *alpha,
                    const double *x, INT incx, double *y, INT incy) {
  return cublasDaxpy(handle, static_cast<int>(n), alpha, x,
      static_cast<int>(incx), y, static_cast<int>(incy));
}

cublasStatus_t axpy(cublasHandle_t handle, INT n, float *alpha,
                    const float *x, INT incx, float *y, INT incy) {
  return cublasSaxpy(handle, static_cast<int>(n), alpha, x,
      static_cast<int>(incx), y, static_cast<int>(incy));
}

// Conjugate Gradient Least Squares.
template <typename T, CGLS_FMT F>
INT solve(cusparseHandle_t handle_s, cublasHandle_t handle_b,
          cusparseMatDescr_t descr, const T *val, const INT *ptr,
          const INT *ind, const INT m, const INT n, const INT nnz, const T *b,
          T *x, const T shift, const T tol, const INT maxit, bool quiet) {

  // Variable declarations.
  T *p, *q, *r, *s;
  T gamma, normp, normq, norms, norms0, normx, xmax;
  char fmt[] = "%5d %9.2e %12.5g\n";
  INT k, flag = 0, indefinite = 0;

  // Constant declarations.
  const T kNegOne = static_cast<T>(-1);
  const T kZero = static_cast<T>(0);
  const T kOne = static_cast<T>(1);
  const T kNegShift = static_cast<T>(-shift);
  const T kEps = static_cast<T>(1e-16);

  // Memory Allocation.
  cudaMalloc(&p, 2 * (m + n) * sizeof(T));
  q = p + n;
  r = q + m;
  s = r + m;
  cudaMemcpy(r, b, m * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);

  // r = b - A*x.
  spmv<T, F>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &kNegOne,
      descr, val, ptr, ind, x, &kOne, r);

  // s = A'*r - shift*x.
  spmv<T, F>(handle_s, CUSPARSE_OPERATION_TRANSPOSE, m, n, nnz, &kOne,
      descr, val, ptr, ind, r, &kNegShift, s);

  // Initialize.
  cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);
  nrm2(handle_b, n, s, 1, &norms);
  cudaDeviceSynchronize();
  norms0 = norms;
  gamma = norms0 * norms0;
  nrm2(handle_b, n, x, 1, &normx);
  cudaDeviceSynchronize();
  xmax = normx;

  if (norms < kEps)
    flag = 1;

  if (!quiet)
    printf("    k     normx        resNE\n");

  for (k = 0; k < maxit && !flag; ++k) {
    // q = A * p.
    spmv<T, F>(handle_s, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &kOne,
        descr, val, ptr, ind, p, &kZero, q);

    // delta = norm(p)^2 + shift*norm(q)^2.
    nrm2(handle_b, n, p, 1, &normp); 
    nrm2(handle_b, m, q, 1, &normq); 
    cudaDeviceSynchronize();
    T delta = normq * normq + shift * normp * normp;

    if (delta <= 0)
      indefinite = 1;
    if(delta == 0)
      delta = kEps;
    T alpha = gamma / delta;
    T neg_alpha = -alpha;

    // x = x + alpha*p.
    // r = r - alpha*q.
    axpy(handle_b, n, &alpha, p, 1, x,  1);
    axpy(handle_b, m, &neg_alpha, q, 1, r,  1);

    // s = A'*r - shift*x.
    cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);
    spmv<T, F>(handle_s, CUSPARSE_OPERATION_TRANSPOSE, m, n, nnz, &kOne,
        descr, val, ptr, ind, r, &kNegShift, s);

    // Compute beta.
    nrm2(handle_b, n, s, 1, &norms);
    cudaDeviceSynchronize();
    T gamma1 = gamma;
    gamma = norms * norms;
    T beta = gamma / gamma1;

    // p = s + beta*p.
    axpy(handle_b, n, &beta, p, 1, s, 1);
    cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);

    // Convergence check.
    nrm2(handle_b, n, x, 1, &normx);
    cudaDeviceSynchronize();
    xmax = std::max(xmax, normx);
    bool converged = (norms <= norms0 * tol) || (normx * tol >= 1);
    if (!quiet && (converged || k % 10 == 0))
      printf(fmt, k, normx, norms / norms0); 
    if (converged)
      break;
  }

  // Determine exit status.
  T shrink = normx / xmax;
  if (k == maxit)
    flag = 2;
  else if (indefinite)
    flag = 3;
  else if (shrink * shrink <= tol)
    flag = 4;

  // Free variables and return;
  cudaFree(p);
  return flag;
}

// CGLS, with pre-initialized cusparseHandle and cublasHandle.
template <typename T, CGLS_FMT F>
INT solve(cusparseMatDescr_t descr, const T *val, const INT *ptr,
          const INT *ind, const INT m, const INT n, const INT nnz,
          const T *b, T *x, const T shift, const T tol, const INT maxit,
          bool quiet) {
  cusparseHandle_t handle_s;
  cublasHandle_t handle_b;
  cusparseCreate(&handle_s);
  cublasCreate(&handle_b);
  return solve<T, F>(handle_s, handle_b, descr, val, ptr, ind, m, n, nnz, b, x,
      shift, tol, maxit, quiet);
}

// CGLS, with pre-initialized cusparseMatDescr, cusparseHandle and cublasHandle.
template <typename T, CGLS_FMT F>
INT solve(const T *val, const INT *ptr, const INT *ind, const INT m,
          const INT n, const INT nnz, const T *b, T *x, const T shift,
          const T tol, const INT maxit, bool quiet) {
  cusparseHandle_t handle_s;
  cublasHandle_t handle_b;
  cusparseCreate(&handle_s);
  cublasCreate(&handle_b);
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);
  return solve<T, F>(handle_s, handle_b, descr, val, ptr, ind, m, n, nnz, b, x,
      shift, tol, maxit, quiet);
}

}  // namespace cgls

#endif  // CGLS_CUH_

