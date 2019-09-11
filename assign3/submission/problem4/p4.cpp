#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <cstring>

using namespace std;

//constants
const char *filename = "input.txt";
const int BuffSize = 8;
const int WriteStride = 4;
const int Pthreads = 1;
const int Cthreads = 2;
int filesize = 24; 
const char *f = "hritvikhritvikhritvikhri";

//buffer and metadata
char buffer[BuffSize] = {0};
int b_start = 0;
int b_end = 0;
int bytes_read = 0;
int bytes_written = 0;

//condition variable 
pthread_cond_t cond_p = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_c = PTHREAD_COND_INITIALIZER;
  
//mutex 
pthread_mutex_t lock_p = PTHREAD_MUTEX_INITIALIZER; 
pthread_mutex_t lock_c = PTHREAD_MUTEX_INITIALIZER; 


void *producer(void *arguments) {
  while(bytes_read != filesize) {
    pthread_mutex_lock(&lock_p);
    while(bytes_read < filesize && (b_end - b_start + BuffSize) % BuffSize == BuffSize - 1) {
      pthread_cond_signal(&cond_c);
      pthread_cond_wait(&cond_p, &lock_p);
    }
    
    if(bytes_read == filesize) {
      pthread_mutex_unlock(&lock_p);
      break;
    }
    
    buffer[b_end] = f[bytes_read];
    bytes_read++;
    b_end = (b_end + 1) % BuffSize;
    
    pthread_mutex_unlock(&lock_p);
    pthread_cond_signal(&cond_c);
  }
  cout << "Producer Exiting" << endl;
  pthread_exit(NULL);
}

void *consumer(void *arguments) {
  int i;
  char out[4];
  while(bytes_written != filesize) {
    pthread_mutex_lock(&lock_c);
    while(bytes_written < filesize && (b_end - b_start + BuffSize) % BuffSize < WriteStride) {
      pthread_cond_signal(&cond_p);
      pthread_cond_wait(&cond_c, &lock_c);
    }
    
    if(bytes_written == filesize) {
      pthread_mutex_unlock(&lock_c);
      break;
    }
  
    i = WriteStride;
    
    while(i--) {
      out[WriteStride - i - 1] = buffer[b_start];
      b_start = (b_start + 1) % BuffSize;
    }
    
    bytes_written += WriteStride;
    pthread_mutex_unlock(&lock_c);
    pthread_cond_signal(&cond_p);
    cerr << out;
  }
  cout << "Consumer Exiting" << endl;
  pthread_cond_signal(&cond_c);
  pthread_exit(NULL);
}

int main() {
  pthread_t p_threads[Pthreads];
  pthread_t c_threads[Cthreads];
  int i, err;
  for(i = 0; i < Pthreads; i++) {
    err = pthread_create(&p_threads[i], NULL, producer, NULL);
    if(err) {
      cout << "ERROR: return code from pthread_create() is " << err << "\n";
      exit(-1);
    }
  }
  
  for(i = 0; i < Cthreads; i++) {
    err = pthread_create(&c_threads[i], NULL, consumer, NULL);
    if(err) {
      cout << "ERROR: return code from pthread_create() is " << err << "\n";
      exit(-1);
    }
  }

  for(i = 0; i < Pthreads; i++) {
    err = pthread_join(p_threads[i], NULL);
    if(err) {
      cout << "ERROR: return code from pthread_join() is " << err << "\n";
      exit(-1);
    }
  }
  
  for(i = 0; i < Cthreads; i++) {
    err = pthread_join(c_threads[i], NULL);
    if(err) {
      cout << "ERROR: return code from pthread_join() is " << err << "\n";
      exit(-1);
    }
  }

}

