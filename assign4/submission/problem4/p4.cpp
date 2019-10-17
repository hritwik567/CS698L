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

void print_queue(int tid, struct list_node_s *queue_head);
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


void do_tokenize(struct list_node_s *node) {
  if(node->data) {
    char delim[] = " ";

	  char *ptr = strtok(node->data, delim);

    while(ptr != NULL)
    {
      #pragma omp critical
      cout << ptr << endl;
      ptr = strtok(NULL, delim);
    }
  }
}

void prod_cons(int prod_count, int cons_count, FILE *files[], int file_count) {
  // SB: Write your OpenMP code here.
  int total_threads = prod_count + cons_count;  

  int f_read = 0;
  int p_done = 0;
  struct list_node_s *p_head = (struct list_node_s*) malloc(sizeof(list_node_s));
  p_head->data = NULL;
  p_head->next = NULL;
  
  int c_done = 0;
  struct list_node_s *c_head = p_head;
  struct list_node_s *t_head = p_head;

  #pragma omp parallel num_threads(total_threads)
  {
    int t_id = omp_get_thread_num();
    if(t_id < file_count) {
      char * line = NULL;
      size_t len = 0;
      size_t read;

      while ((read = getline(&line, &len, files[t_id])) != -1) {
        struct list_node_s *t_node = (struct list_node_s*) malloc(sizeof(list_node_s));
        t_node->data = (char*) malloc((read - 1) * sizeof(char));
        t_node->next = NULL;
        strncpy(t_node->data, line, read - 1);
        #pragma omp critical
        {
          p_head->next = t_node;
          p_head = p_head->next;
          p_done += 1;
        }
      }
      #pragma omp critical
      f_read += 1;
      free(line);
    } else {
      struct list_node_s *l_head;
      while(true) {
        while(c_done == p_done and f_read != file_count);
        #pragma omp critical
        {
          if(c_head->next != NULL) {
            c_head = c_head->next;
            c_done += 1;
            l_head = c_head;
          } else {
            l_head = NULL;
          }
        }
        if(l_head) do_tokenize(l_head);
        else break;
      }
    }
  }
}

void print_queue(int tid, struct list_node_s *queue_head) {
  cout << "Thread: " << tid << " -> queue = \n";
  struct list_node_s *curr_p = queue_head;
#pragma omp critical
  while (curr_p != NULL) {
    if(curr_p->data)
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
