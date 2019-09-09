// Compile: g++ -O3 -o example1 example1.cpp
// Execute: ./example1

#include <iostream>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

const int N = 4096;
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

void reference(double** A, double* x, double* y_ref, double* z_ref) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(double* w_ref, double* w_opt) {
  double maxdiff, this_diff;
  int numdiffs;
  int i;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (this_diff < 0)
      this_diff = -1.0 * this_diff;
    if (this_diff > threshold) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << threshold
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THIS CODE
void optimized(double **A, double *x, double *y_opt, double *z_opt) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}

int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(3);

  double **A;
  A = new double *[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (j = 0; j < N; j++)
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    reference(A, x, y_ref, z_ref);
  clkend = rtclock();
  t = clkend - clkbegin;
  if (y_ref[N / 2] * y_ref[N / 2] < -100.0)
    cout << y_ref[N / 2] << "\n";
  cout << "Reference Version: Matrix Size = " << N << ", "
       << 4.0 * 1e-9 * N * N * Niter / t << " GFLOPS; Time = " << t << " sec\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized(A, x, y_opt, z_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  if (y_opt[N / 2] * y_opt[N / 2] < -100.0)
    cout << y_opt[N / 2] << "\n";
  cout << "Optimized Version: Matrix Size = " << N << ", "
       << 4.0 * 1e-9 * N * N * Niter / t << " GFLOPS; Time = " << t << " sec\n";

  check_result(y_ref, y_opt);
}
