#include <stdlib.h>
#include <cmath>

#include "cgls.cuh"

// Define real type.
typedef double real_t;

// Generates random CSR matrix with entries in [-1, 1]. The matrix will have
// exactly nnz non-zeros. All arrays must be pre-allocated.
void CsrMatGen(int m, int n, int nnz, real_t *val, int *r_ptr, int *c_ind) {
  real_t kRandMax = static_cast<real_t>(RAND_MAX);
  real_t kM = static_cast<real_t>(m);
  real_t kN = static_cast<real_t>(n);

  int num = 0;
  for (int i = 0; i < m; ++i) {
    r_ptr[i] = num;
    for (int j = 0; j < n; ++j) {
      if (rand() / kRandMax * ((kM - i) * kN - j) < (nnz - num - 1)) {
        val[num] = 2 * (rand() - kRandMax / 2) / kRandMax;
        c_ind[num] = j;
        num++;
      }
    }
  }
  r_ptr[m] = nnz;
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
  int c_ind_h[]   = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
  int r_ptr_h[]   = {0, 3, 5, 8, 11, 13};
  real_t b_h[]    = {-2, -1,  0,  1,  2};
  real_t x_h[]    = {0,  0,  0,  0,  0};
  real_t x_star[] = { 0.461620337853983,  0.025458521291462, -0.509793131412600,
                      0.579159637092979, -0.350590484189795};

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *c_ind_d, *r_ptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&c_ind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  r_ptr_d = c_ind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(c_ind_d, c_ind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(r_ptr_d, r_ptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve.
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, r_ptr_d, c_ind_d, m, n,
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
  cudaFree(c_ind_d);
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
  int c_ind_h[] = {0, 1, 2, 3, 2, 3, 3, 4, 0, 1, 0, 2, 3, 1, 4, 4, 0};
  int r_ptr_h[] = {0, 4, 6, 6, 8, 10, 13, 14, 15, 16, 17};
  real_t b_h[]    = { 1.340034585145723, -0.634242023306306, -0.213297722346186,
                     -0.129598039513105,  0.132020354623637,  0.078143427011308,
                      0.300482010299278, -0.688536305275490, -0.465698657933079,
                      0.074768275950993};
  real_t x_h[]    = {0, 0, 0, 0, 0};
  real_t x_star[] = { 0.066707422952301,  0.308024162523591, -0.843805757764051,
                     -1.276669375807300,  0.577067691426442};

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *c_ind_d, *r_ptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&c_ind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  r_ptr_d = c_ind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(c_ind_d, c_ind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(r_ptr_d, r_ptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve.
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, r_ptr_d, c_ind_d, m, n,
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
  cudaFree(c_ind_d);
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
  int *c_ind_h = new int[nnz];
  int *r_ptr_h = new int[m + 1];
  real_t *b_h = new real_t[m];
  real_t *x_h = new real_t[n]();

  // Generate data.
  CsrMatGen(m, n, nnz, val_h, r_ptr_h, c_ind_h);
  for (int i = 0; i < m; ++i)
    b_h[i] = rand() / static_cast<real_t>(RAND_MAX);

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *c_ind_d, *r_ptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&c_ind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  r_ptr_d = c_ind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(c_ind_d, c_ind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(r_ptr_d, r_ptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve.
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, r_ptr_d, c_ind_d, m, n,
      nnz, b_d, x_d, shift, tol, maxit, quiet);

  // Check Result
  if (flag == 0)
    printf("Test3 Passed: Flag = %d\n", flag);
  else
    printf("Test3 Failed: Flag = %d\n", flag);

  // Free data.
  cudaFree(val_d);
  cudaFree(c_ind_d);
  delete [] val_h;
  delete [] r_ptr_h;
  delete [] c_ind_h;
  delete [] x_h;
  delete [] b_h;
}

// Run tests.
int main() {
  test1();
  test2();
  test3();
}

