#ifndef _UTILS_H_
#define _UTILS_H_

#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iostream>

#define MEMORY_ALIGNMENT 64

class Timer {
  public:

    void tic() {
      t_start = std::chrono::high_resolution_clock::now();
    }

    double toc() {
      auto t_end = std::chrono::high_resolution_clock::now();
      return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
};

template <class ValueType> ValueType read_option(const char* option, int argc, char** argv, const char* default_value = nullptr);

template <> std::string read_option<std::string>(const char* option, int argc, char** argv, const char* default_value) {
  for (int i = 0; i < argc - 1; i++) {
    if (!strcmp(argv[i], option)) {
      return std::string(argv[i+1]);
    }
  }
  if (default_value) return std::string(default_value);
  std::cerr<<"Option "<<option<<" was not provided. Exiting...\n";
  exit(1);
}
template <> int read_option<int>(const char* option, int argc, char** argv, const char* default_value) {
  return strtol(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL, 10);
}
template <> long read_option<long>(const char* option, int argc, char** argv, const char* default_value) {
  return strtol(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL, 10);
}
template <> float read_option<float>(const char* option, int argc, char** argv, const char* default_value) {
  return strtod(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL);
}
template <> double read_option<double>(const char* option, int argc, char** argv, const char* default_value) {
  return strtof(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL);
}

inline void* aligned_malloc(size_t size) {
  constexpr uintptr_t ALIGN_MASK = MEMORY_ALIGNMENT - 1;
  char* p = (char*)malloc(size + 2 + ALIGN_MASK);
  if (!p) return p;

  char* p_aligned = (char*)((uintptr_t)(p + 2 + ALIGN_MASK) & ~(uintptr_t)ALIGN_MASK);
  ((uint16_t*)p_aligned)[-1] = (uint16_t)(p_aligned - p);
  return p_aligned;
}

inline void aligned_free(void* p_aligned) {
  if (p_aligned) {
    char* p = ((char*)p_aligned) - ((uint16_t*)p_aligned)[-1];
    free(p);
  }
}

#endif
