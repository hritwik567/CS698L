// Compile:  g++ -fopenmp assignment4-p4.cpp -o assignment4-p4
// Run: ./assignment4-p4 <producers> <consumers>

#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

const int MAX_FILES = 10;
const int MAX_CHAR = 100;

using namespace std;

// struct for queue
struct list_node_s {
  char *data;
  struct list_node_s *next;
};

void print_usage(char *prog_name) {
  cerr << "usage: " << prog_name << " <producer count> <consumer count>\n";
  exit(EXIT_FAILURE);
}

// Read in list of filenames and open files
void get_files(FILE *files[], int *file_count_p, int prod_count) {
  int i = 0;
  char filename[MAX_CHAR];
  while (1) {
    scanf("%s", filename);
    files[i] = fopen(filename, "r");
    if (files[i] == NULL) {
      cout << "Cannot open " << filename << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
    if (i == prod_count) {
      break;
    }
  }
  *file_count_p = i;
}

void prod_cons(int prod_count, int cons_count, FILE *files[], int file_count) {
   // SB: Write your OpenMP code here.
}

void print_queue(int tid, struct list_node_s *queue_head) {
  cout << "Thread: " << tid << " -> queue = \n";
  struct list_node_s *curr_p = queue_head;
#pragma omp critical
  while (curr_p != NULL) {
    cout << curr_p->data << "\n";
    curr_p = curr_p->next;
  }
  cout << "\n";
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    print_usage(argv[0]);
  }

  int prod_count = strtol(argv[1], NULL, 10);
  int cons_count = strtol(argv[2], NULL, 10);

  FILE *files[MAX_FILES];
  int file_count;
  get_files(files, &file_count, prod_count);

  // Producer-consumer
  prod_cons(prod_count, cons_count, files, file_count);

  return 0;
}
