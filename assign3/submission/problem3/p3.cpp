#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <cstring>

using namespace std;

int N = 1024;
int Nthreads = 8;
const int Niter = 1;

struct thr_args {
  int **A;
  int **B;
  int **C;
  int start;
  int offset;
};

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

void sequential(int **A, int **B, int **C) {
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void *parallel_s1(void *arguments) {
  struct thr_args *tmp = static_cast<struct thr_args*>(arguments);
  int **A = tmp->A, **B = tmp->B, **C = tmp->C;
  int start = tmp->start, offset = tmp->offset;
  int i, j, k;
  for (i = start; i < N; i += offset) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  pthread_exit(NULL);
}

void parallel(int **A, int **B, int **C) {
  pthread_t *threads = new pthread_t[Nthreads];
  int i, err;
  struct thr_args args[Nthreads] = {0};
  for(i = 0; i < Nthreads; i++) {
    args[i].A = A; args[i].B = B; args[i].C = C;
    args[i].start = i;
    args[i].offset = Nthreads;
    err = pthread_create(&threads[i], NULL, parallel_s1, args + i);
    if(err) {
      cout << "ERROR: return code from pthread_create() is " << err << "\n";
      exit(-1);
    }
  }
  
  for(i = 0; i < Nthreads; i++) {
    err = pthread_join(threads[i], NULL);
    if(err) {
      cout << "ERROR: return code from pthread_join() is " << err << "\n";
      exit(-1);
    }
  }
}

void check_result(int **C_seq, int **C_para) {
  int numdiffs;
  int i, j;
  numdiffs = 0;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (C_seq[i][j] != C_para[i][j]) {
        numdiffs++;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found\n";
  } else {
    cout << "No differences found between sequential and parallel versions\n";
  }
}

int main(int argc, char **argv) {
  if(argc != 1 && argc != 3) {
    cout << "Usage ./binary <MatrixSize> <NumberThreads>" << endl;
    exit(-1);
  } else if(argc == 3) {
    N = stoi(argv[1]);
    Nthreads = stoi(argv[2]);
  }
  
  double clkbegin, clkend;
  double t_seq, t_para;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  int **A, **B, **C_seq, **C_para;
  A = new int *[N];
  for (i = 0; i < N; i++) {
    A[i] = new int[N];
    memset(A[i], 1, N * sizeof(int));
  }

  B = new int *[N];
  for (i = 0; i < N; i++) {
    B[i] = new int[N];
    memset(B[i], 1, N * sizeof(int));
  }

  C_seq = new int *[N];
  C_para = new int *[N];
  for (i = 0; i < N; i++) {
    C_seq[i] = new int[N]();
    C_para[i] = new int[N]();
  }
  
  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    sequential(A, B, C_seq);
  clkend = rtclock();
  t_seq = clkend - clkbegin;
  
  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    parallel(A, B, C_para);
  clkend = rtclock();
  t_para = clkend - clkbegin;
  
  cout << "Matrix Size = " << N << "\n"
       << "Number of Threads = " << Nthreads << "\n"
       << "Number of iterations = " << Niter << "\n"
       << "Sequential Time = " << t_seq / Niter
       << "sec\n"
       << "Parallel Time = " << t_para / Niter
       << "sec\n"
       << "Speedup = " << t_seq / t_para << "\n";
  check_result(C_seq, C_para);
}
