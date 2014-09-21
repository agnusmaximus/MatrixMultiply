#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#include "smmintrin.h"
#include "emmintrin.h"
#include "immintrin.h"

using namespace std;

#define DIM 1024
#define PRINT 0
#define N_THREADS 4
#define LINE_SIZE 32
#define BENCHMARK_NAIVE 0
#define BENCHMARK_SIMPLE_PAR 0
#define BENCHMARK_BLOCKED 0
#define BENCHMARK_BLOCKED_PAR 1
#define N_BENCHMARK_ITER 50
#define VERIFY 0

void mat_mul_aligned(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM; ++j) {
      for (int k = 0; k < DIM; ++k) {
        c[i*DIM+j] += a[i*DIM+k] * b[k*DIM+j];
      }
    }
  }
}

void mat_mul_parallel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  omp_set_num_threads(N_THREADS);
#pragma omp parallel for  
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM; ++j) {
      for (int k = 0; k < DIM; k+=8) {
        c[i*DIM+j] += a[i*DIM+k] * b[k*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+1] * b[(k+1)*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+2] * b[(k+2)*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+3] * b[(k+3)*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+4] * b[(k+4)*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+5] * b[(k+5)*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+6] * b[(k+6)*DIM+j];
        c[i*DIM+j] += a[i*DIM+k+7] * b[(k+7)*DIM+j];
      }
    }
  }
}

void mat_mul_subblock_AVX(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  for (int i = 0; i < LINE_SIZE; ++i) {
    __m256 c1 = _mm256_load_ps((const float *)&c[i*DIM]);
    __m256 c2 = _mm256_load_ps((const float *)&c[i*DIM+8]);
    __m256 c3 = _mm256_load_ps((const float *)&c[i*DIM+16]);
    __m256 c4 = _mm256_load_ps((const float *)&c[i*DIM+24]);

    for (int k = 0; k < LINE_SIZE; ++k) {
      __m256 b1 = _mm256_load_ps((const float *)&b[k*DIM]);
      __m256 b2 = _mm256_load_ps((const float *)&b[k*DIM+8]);
      __m256 b3 = _mm256_load_ps((const float *)&b[k*DIM+16]);
      __m256 b4 = _mm256_load_ps((const float *)&b[k*DIM+24]);
      __m256 a1 = _mm256_set1_ps(a[i*DIM+k]);
      
      c1 = _mm256_fmadd_ps(a1, b1, c1);
      c2 = _mm256_fmadd_ps(a1, b2, c2);
      c3 = _mm256_fmadd_ps(a1, b3, c3);
      c4 = _mm256_fmadd_ps(a1, b4, c4);
    }
    
    _mm256_store_ps(&c[i*DIM], c1);
    _mm256_store_ps(&c[i*DIM+8], c2);
    _mm256_store_ps(&c[i*DIM+16], c3);
    _mm256_store_ps(&c[i*DIM+24], c4);
  }
}

void mat_mul_subblock_SSE4(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  for (int i = 0; i < LINE_SIZE; ++i) {
    __m128 c1 = _mm_load_ps((const float *)&c[i*DIM]);
    __m128 c2 = _mm_load_ps((const float *)&c[i*DIM+4]);
    __m128 c3 = _mm_load_ps((const float *)&c[i*DIM+8]);
    __m128 c4 = _mm_load_ps((const float *)&c[i*DIM+12]);
    __m128 c5 = _mm_load_ps((const float *)&c[i*DIM+16]);
    __m128 c6 = _mm_load_ps((const float *)&c[i*DIM+20]);
    __m128 c7 = _mm_load_ps((const float *)&c[i*DIM+24]);
    __m128 c8 = _mm_load_ps((const float *)&c[i*DIM+28]);
    for (int k = 0; k < LINE_SIZE; ++k) {
      __m128 b1 = _mm_load_ps((const float *)&b[k*DIM]);
      __m128 b2 = _mm_load_ps((const float *)&b[k*DIM+4]);
      __m128 b3 = _mm_load_ps((const float *)&b[k*DIM+8]);
      __m128 b4 = _mm_load_ps((const float *)&b[k*DIM+12]);
      __m128 b5 = _mm_load_ps((const float *)&b[k*DIM+16]);
      __m128 b6 = _mm_load_ps((const float *)&b[k*DIM+20]);
      __m128 b7 = _mm_load_ps((const float *)&b[k*DIM+24]);
      __m128 b8 = _mm_load_ps((const float *)&b[k*DIM+28]);
      __m128 a1 = _mm_set1_ps(a[i*DIM+k]);

      c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
      c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b2));
      c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b3));
      c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b4));
      c5 = _mm_add_ps(c5, _mm_mul_ps(a1, b5));
      c6 = _mm_add_ps(c6, _mm_mul_ps(a1, b6));
      c7 = _mm_add_ps(c7, _mm_mul_ps(a1, b7));
      c8 = _mm_add_ps(c8, _mm_mul_ps(a1, b8));
    }
    _mm_store_ps(&c[i*DIM], c1);
    _mm_store_ps(&c[i*DIM+4], c2);
    _mm_store_ps(&c[i*DIM+8], c3);
    _mm_store_ps(&c[i*DIM+12], c4);
    _mm_store_ps(&c[i*DIM+16], c5);
    _mm_store_ps(&c[i*DIM+20], c6);
    _mm_store_ps(&c[i*DIM+24], c7);
    _mm_store_ps(&c[i*DIM+28], c8);
  }
}

void mat_mul_block(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  int range = DIM/LINE_SIZE;
  for (int i = 0; i < DIM; i+=LINE_SIZE) {
    for (int j = 0; j < DIM; j+=LINE_SIZE) {
      for (int k = 0; k < DIM; k+=LINE_SIZE) {
        mat_mul_subblock_AVX(a+i*DIM+k, b+k*DIM+j, c+i*DIM+j);
      }
    }
  }
}

void mat_mul_block_par(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
  int range = DIM/LINE_SIZE;
  omp_set_num_threads(N_THREADS);
#pragma omp parallel for
  for (int i = 0; i < DIM; i+=LINE_SIZE) {
    for (int j = 0; j < DIM; j+=LINE_SIZE) {
      for (int k = 0; k < DIM; k+=LINE_SIZE) {
        mat_mul_subblock_AVX(a+i*DIM+k, b+k*DIM+j, c+i*DIM+j);
      }
    }
  }
}

long long int get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void check(float * __restrict__ a, float * __restrict__ b) {
  for (int i = 0; i < DIM * DIM; ++i) {
    if (a[i] != b[i]) {
      cout << "Error, not the same" << endl;
      return;
    }
  }
  cout << "All good" << endl;
}

double report_mflops(double t) {
  double n = DIM;
  double f = (2.0*(n*n*n) / t) / 1000000;
  return f;
}

void print_mat(float * __restrict__ a) {
  if (!PRINT) return;
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM; ++j) {
      cout << a[i*DIM+j] << " ";
    }
    cout << endl;
  }
  cout << "----------------------" << endl;
}

void benchmark(void (*mm)(float *, float *, float*), string name) {
  
  double theoretical_max = 38400;

  clock_t avg_time = 0;
  float avg_mflops = 0;
  
  for (int i = 0; i < N_BENCHMARK_ITER; ++i) {
    float * a __attribute__((aligned(16))) = new float[DIM*DIM];
    float * b __attribute__((aligned(16))) = new float[DIM*DIM];
    
    for (int i = 0; i < DIM*DIM; ++i) {
      a[i] = rand() % 1000;
      b[i] = rand() % 1000;
    }
  
    float *c __attribute__((aligned(16))) = new float[DIM*DIM];
    clock_t t = get_time();
    mm(a, b, c);
    t = get_time() - t;
    
    avg_time += t;
    avg_mflops += report_mflops((double)t/1000);

    if (VERIFY) {
      float *cv __attribute__((aligned(16))) = new float[DIM*DIM];
      mat_mul_aligned(a, b, cv);
      check(c, cv);
    }
  }

  avg_time /= N_BENCHMARK_ITER;
  avg_mflops /= N_BENCHMARK_ITER;
  cout << name << "_MFlops: " << avg_mflops << "| Time: " << avg_time 
       << "| Efficiency: " << avg_mflops / theoretical_max << endl;
}

int main(int argc, char * argv[]) {
  srand(time(NULL));

  if (BENCHMARK_NAIVE) {
    benchmark(mat_mul_aligned, "Simple");
  }
  if (BENCHMARK_SIMPLE_PAR) {
    benchmark(mat_mul_parallel, "Parallel");
  }
  if (BENCHMARK_BLOCKED) {
    benchmark(mat_mul_block, "Blocked");
  }
  if (BENCHMARK_BLOCKED_PAR) {
    benchmark(mat_mul_block_par, "Parallel_Blocked");
  }
}
