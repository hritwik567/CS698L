#include <iostream>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

const int N = 1024;
const int Niter = 10;
const double threshold = 0.0000001;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double **A, double **B, double **C) {
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < i + 1; k++) {
        C[i][j] += A[k][i] * B[j][k];
      }
    }
  }
}

void check_result(double **w_ref, double **w_opt) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (this_diff < 0)
        this_diff = -1.0 * this_diff;
      if (this_diff > threshold) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << threshold
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// THIS IS INITIALLY IDENTICAL TO REFERENCE
// MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
void optimized(double **A, double **B, double **C) {
  int i, j, k, s = 8, ii, jj, kk;
  double r;
  for (i = 0; i < N; i += s) {
    for (j = 0; j < N; j += s) {
      for(ii = i; ii < min(i + s, N); ii++) {
        for(jj = j; jj < min(j + s, N); jj++) {
          r = B[ii][jj];
          for(k = jj; k < N; k += s) {
              C[k][ii] += A[jj][k] * r;
          }
        }
      }
    }
  }
}

int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double **A, **B, **C_ref, **C_opt;
  A = new double *[N];
  for (i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  B = new double *[N];
  for (i = 0; i < N; i++) {
    B[i] = new double[N];
  }

  C_ref = new double *[N];
  for (i = 0; i < N; i++) {
    C_ref[i] = new double[N];
  }

  C_opt = new double *[N];
  for (i = 0; i < N; i++) {
    C_opt[i] = new double[N];
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = i + j + 1;
      A[i][j] = random();
      B[i][j] = (i + 1) * (j + 1);
      B[i][j] = random();
      C_ref[i][j] = 0.0;
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    reference(A, B, C_ref);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", "
       << 2.0 * 1e-9 * N * N * Niter / t << " GFLOPS; Time = " << t / Niter
       << " sec\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", "
       << 2.0 * 1e-9 * N * N * Niter / t << " GFLOPS; Time = " << t / Niter
       << " sec\n";

  check_result(C_ref, C_opt);
}
