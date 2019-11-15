// Compile: g++ -std=c++11 assignment5-p1.cpp -ltbb -o assignment5-p1

#include <cassert>
#include <chrono>
#include <iostream>
#include <tbb/tbb.h>

using namespace std;
using namespace std::chrono;
using namespace tbb;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (1 << 24)

uint32_t SerialMaxIndex(const uint32_t* a) {
  uint32_t value_of_max = 0;
  uint32_t index_of_max = -1;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t max_idx = SerialMaxIndex(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << max_idx << " in " << duration << " us\n";

  start = HR::now();
  // SB: Implement a parallel max function with Intel TBB
  end = HR::now();

  // SB: Assert that the two indices computed are the same

  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel max index in " << duration << " us\n";

  return EXIT_SUCCESS;
}
