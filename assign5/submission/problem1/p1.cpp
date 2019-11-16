// Compile: g++ -std=c++11 assignment5-p1.cpp -ltbb -o assignment5-p1

#include <cassert>
#include <chrono>
#include <iostream>
#include <tbb/tbb.h>
#include <tbb/parallel_reduce.h>

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

uint32_t ParallelMaxIndex(const uint32_t* a) {

  pair<uint32_t, uint32_t> id(-1, 0);
  pair<uint32_t, uint32_t> ret = parallel_reduce(blocked_range<uint32_t>(0, N), id, 
    [&](const blocked_range<uint32_t> &r, pair<uint32_t, uint32_t> v) -> pair<uint32_t, uint32_t> {
      for(uint32_t i=r.begin(); i!=r.end(); ++i) {
        if(a[i] > v.second) {
          v.second = a[i];
          v.first = i;
        }
      }
      return v;
    },
    [](pair<uint32_t, uint32_t> u, pair<uint32_t, uint32_t> v) -> pair<uint32_t, uint32_t> {
      if(u.second > v.second) return u;
      if(u.second < v.second) return v;
      if(u.first < v.first) return u;
      return v;
    }
  );
  return ret.first;
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = SerialMaxIndex(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us\n";

  start = HR::now();
  // SB: Implement a parallel max function with Intel TBB
  uint64_t p_max_idx = ParallelMaxIndex(a);
  end = HR::now();

  // SB: Assert that the two indices computed are the same
  assert(s_max_idx == p_max_idx);

  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel max index: " << p_max_idx << " in " << duration << " us\n";

  return EXIT_SUCCESS;
}
