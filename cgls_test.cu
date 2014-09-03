#include <stdlib.h>
#include <cmath>

#include "cgls.cuh"

// Define real type.
typedef double real_t;
#define csr2csc cusparseDcsr2csc

// Generates random CSR matrix with entries in [-1, 1]. The matrix will have
// exactly nnz non-zeros. All arrays must be pre-allocated.
void CsrMatGen(int m, int n, int nnz, real_t *val, int *rptr, int *cind) {
  real_t kRandMax = static_cast<real_t>(RAND_MAX);
  real_t kM = static_cast<real_t>(m);
  real_t kN = static_cast<real_t>(n);

  int num = 0;
  for (int i = 0; i < m; ++i) {
    rptr[i] = num;
    for (int j = 0; j < n && num < nnz; ++j) {
      if (rand() / kRandMax * ((kM - i) * kN - j) < (nnz - num)) {
        val[num] = 2 * (rand() - kRandMax / 2) / kRandMax;
        cind[num] = j;
        num++;
      }
    }
  }
  rptr[m] = nnz;
}

// Test CGLS on square system of equations with known solution.
void test1() {
  // Initialize variables.
  real_t shift = 1;
  real_t tol = 1e-6;
  int maxit = 20;
  bool quiet = false;
  int m = 5;
  int n = 5;
  int nnz = 13;

  // Initialize data.
  real_t val_h[]  = { 1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5};
  int cind_h[]   = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
  int rptr_h[]   = {0, 3, 5, 8, 11, 13};
  real_t b_h[]    = {-2, -1,  0,  1,  2};
  real_t x_h[]    = {0,  0,  0,  0,  0};
  real_t x_star[] = { 0.461620337853983,  0.025458521291462, -0.509793131412600,
                      0.579159637092979, -0.350590484189795};

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *cind_d, *rptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&cind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  rptr_d = cind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cind_d, cind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(rptr_d, rptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve.
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, rptr_d, cind_d, m, n,
      nnz, b_d, x_d, shift, tol, maxit, quiet);

  // Retrieve solution.
  cudaMemcpy(x_h, x_d, n * sizeof(real_t), cudaMemcpyDeviceToHost);

  // Compute error and print.
  real_t err = 0;
  for (int i = 0; i < n; ++i)
    err += (x_h[i] - x_star[i]) * (x_h[i] - x_star[i]);
  err = std::sqrt(err);
  if (flag == 0 && err < tol)
    printf("Test1 Passed: Flag = %d, Error = %e\n", flag, err);
  else
    printf("Test1 Failed: Flag = %d, Error = %e\n", flag, err);

  // Free data.
  cudaFree(val_d);
  cudaFree(cind_d);
}

// Test CGLS on rectangular system of equations with known solution.
void test2() {
  // Initialize variables.
  real_t shift = 0.1;
  real_t tol = 1e-6;
  int maxit = 20;
  bool quiet = false;
  int m = 10;
  int n = 5;
  int nnz = 17;

  // Initialize data.
  real_t val_h[]  = { 0.503206792576615, -0.064342931468363,  0.273525398508089,
                     -0.876171296658172,  0.699498416627245,  0.006382734094307,
                     -0.872461490857631, -1.927164633937109, -1.655186057400025,
                      0.140300920195852,  0.745416695810262, -0.949513158012032,
                      0.753179647233809,  0.117556530400676, -1.458256332931324,
                     -0.742412119936071, -0.269214611464301};
  int cind_h[] = {0, 1, 2, 3, 2, 3, 3, 4, 0, 1, 0, 2, 3, 1, 4, 4, 0};
  int rptr_h[] = {0, 4, 6, 6, 8, 10, 13, 14, 15, 16, 17};
  real_t b_h[]    = { 1.340034585145723, -0.634242023306306, -0.213297722346186,
                     -0.129598039513105,  0.132020354623637,  0.078143427011308,
                      0.300482010299278, -0.688536305275490, -0.465698657933079,
                      0.074768275950993};
  real_t x_h[]    = {0, 0, 0, 0, 0};
  real_t x_star[] = { 0.066707422952301,  0.308024162523591, -0.843805757764051,
                     -1.276669375807300,  0.577067691426442};

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *cind_d, *rptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&cind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  rptr_d = cind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cind_d, cind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(rptr_d, rptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve.
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, rptr_d, cind_d, m, n,
      nnz, b_d, x_d, shift, tol, maxit, quiet);

  // Retrieve solution.
  cudaMemcpy(x_h, x_d, n * sizeof(real_t), cudaMemcpyDeviceToHost);

  // Compute error and print.
  real_t err = 0;
  for (int i = 0; i < n; ++i)
    err += (x_h[i] - x_star[i]) * (x_h[i] - x_star[i]);
  err = std::sqrt(err);
  if (flag == 0 && err < tol)
    printf("Test2 Passed: Flag = %d, Error = %e\n", flag, err);
  else
    printf("Test2 Failed: Flag = %d, Error = %e\n", flag, err);

  // Free data.
  cudaFree(val_d);
  cudaFree(cind_d);
}

// Test CGLS on larger random matrix.
void test3() {
  // Initialize variables.
  real_t shift = 1;
  real_t tol = 1e-6;
  int maxit = 100;
  bool quiet = false;
  int m = 100;
  int n = 1000;
  int nnz = 10000;

  // Initialize data.
  real_t *val_h = new real_t[nnz];
  int *cind_h = new int[nnz];
  int *rptr_h = new int[m + 1];
  real_t *b_h = new real_t[m];
  real_t *x_h = new real_t[n]();

  // Generate data.
  CsrMatGen(m, n, nnz, val_h, rptr_h, cind_h);
  for (int i = 0; i < m; ++i)
    b_h[i] = rand() / static_cast<real_t>(RAND_MAX);

  // Transfer variables to device.
  real_t *val_a_d, *b_d, *x_d;
  int *cind_a_d, *rptr_a_d;

  cudaMalloc(&val_a_d, nnz * sizeof(real_t));
  cudaMalloc(&x_d, n * sizeof(real_t));
  cudaMalloc(&b_d, m * sizeof(real_t));
  cudaMalloc(&cind_a_d, nnz * sizeof(int));
  cudaMalloc(&rptr_a_d, (m + 1) * sizeof(int));

  cudaMemcpy(val_a_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cind_a_d, cind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(rptr_a_d, rptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Make A^T copy.
  real_t *val_at_d;
  int *cind_at_d, *rptr_at_d;

  cudaMalloc(&val_at_d, nnz * sizeof(real_t));
  cudaMalloc(&cind_at_d, nnz * sizeof(int));
  cudaMalloc(&rptr_at_d, (n + 1) * sizeof(int));
  rptr_at_d = cind_at_d + nnz;

  cusparseHandle_t handle_s;
  cusparseCreate(&handle_s);
  csr2csc(handle_s, m, n, nnz, val_a_d, rptr_a_d, cind_a_d, val_at_d,
      cind_at_d, rptr_at_d, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO);
  cudaDeviceSynchronize();
  cusparseDestroy(handle_s);

  // Solve.
  int flag = cgls::solve<real_t, cgls::CSR>(val_a_d, rptr_a_d, cind_a_d,
      val_at_d, rptr_at_d, cind_at_d, m, n, nnz, b_d, x_d, shift, tol, maxit,
      quiet);

  // Check Result
  if (flag == 0)
    printf("Test3 Passed: Flag = %d\n", flag);
  else
    printf("Test3 Failed: Flag = %d\n", flag);

  // Free data.
  cudaFree(val_a_d);
  cudaFree(cind_a_d);
  cudaFree(val_at_d);
  cudaFree(cind_at_d);
  delete [] val_h;
  delete [] rptr_h;
  delete [] cind_h;
  delete [] x_h;
  delete [] b_h;
}

// Run tests.
int main() {
  test1();
  test2();
  test3();
}

