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

void mat_mul_aligned(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      for (int k = 0; k < DIM; k++) {
        c[i*DIM+j] += a[i*DIM+k] * b[k*DIM+j];
      }
    }
  }
}

void mat_mul_parallel(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  omp_set_num_threads(N_THREADS);
#pragma omp parallel for  
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
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

void mat_mul_subblock(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  for (int i = 0; i < LINE_SIZE; i++) {
    __m256i c1 = _mm256_load_si256((const __m256i *)&c[i*DIM]);
    __m256i c2 = _mm256_load_si256((const __m256i *)&c[i*DIM+8]);
    __m256i c3 = _mm256_load_si256((const __m256i *)&c[i*DIM+16]);
    __m256i c4 = _mm256_load_si256((const __m256i *)&c[i*DIM+24]);
    for (int k = 0; k < LINE_SIZE; k++) {
      __m256i b1 = _mm256_load_si256((const __m256i *)&b[k*DIM]);
      __m256i b2 = _mm256_load_si256((const __m256i *)&b[k*DIM+8]);
      __m256i b3 = _mm256_load_si256((const __m256i *)&b[k*DIM+16]);
      __m256i b4 = _mm256_load_si256((const __m256i *)&b[k*DIM+24]);
      __m256i a1 = _mm256_set1_epi32(a[i*DIM+k]);
      
      c1 = _mm256_add_epi32(c1, _mm256_mullo_epi32(a1, b1));
      c2 = _mm256_add_epi32(c2, _mm256_mullo_epi32(a1, b2));
      c3 = _mm256_add_epi32(c3, _mm256_mullo_epi32(a1, b3));
      c4 = _mm256_add_epi32(c4, _mm256_mullo_epi32(a1, b4));
      
      _mm256_store_si256((__m256i *)&c[i*DIM], c1);
      _mm256_store_si256((__m256i *)&c[i*DIM+8], c2);
      _mm256_store_si256((__m256i *)&c[i*DIM+16], c3);
      _mm256_store_si256((__m256i *)&c[i*DIM+24], c4);
    }
  }
}

void mat_mul_block(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  int range = DIM/LINE_SIZE;
  for (int i = 0; i < DIM; i+=LINE_SIZE) {
    for (int j = 0; j < DIM; j+=LINE_SIZE) {
      for (int k = 0; k < DIM; k+=LINE_SIZE) {
        mat_mul_subblock(a+i*DIM+k, b+k*DIM+j, c+i*DIM+j);
      }
    }
  }
}

void reorder(int * __restrict__ a, int * __restrict__ b) {
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      b[i*DIM+j] = a[j*DIM+i];
    }
  }
}

void mat_mul_block2(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  int *b2 = new int[DIM*DIM];
  reorder(b, b2);
  for (int i = 0; i < DIM; i+=LINE_SIZE) {
    for (int j = 0; j < DIM; j+=LINE_SIZE) {
      for (int k = 0; k < DIM; k+=LINE_SIZE) {
        mat_mul_subblock(a+i*DIM+k, b2+j*DIM+k, c+i*DIM+j);
        //c[i*DIM+j] += a[i*DIM+k] * b2[j*DIM+k];    
      }
    }
  }
}

void mat_mul_block_par(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  int range = DIM/LINE_SIZE;
#pragma omp parallel for
  for (int i = 0; i < DIM; i+=LINE_SIZE) {
    for (int j = 0; j < DIM; j+=LINE_SIZE) {
      for (int k = 0; k < DIM; k+=LINE_SIZE) {
        mat_mul_subblock(a+i*DIM+k, b+k*DIM+j, c+i*DIM+j);
      }
    }
  }
}

long long int get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void check(int * __restrict__ a, int * __restrict__ b) {
  for (int i = 0; i < DIM * DIM; i++) {
    if (a[i] != b[i]) {
      cout << "Error, not the same" << endl;
      break;
    }
  }
}

double report_mflops(double t) {
  double n = DIM;
  double f = (2.0*(n*n*n) / t) / 1000000;
  return f;
}

void print_mat(int * __restrict__ a) {
  if (!PRINT) return;
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      cout << a[i*DIM+j] << " ";
    }
    cout << endl;
  }
  cout << "----------------------" << endl;
}

int main(int argc, char * __restrict__ argv[]) {
  srand(time(NULL));

  int * a = new int[DIM*DIM];
  int * b = new int[DIM*DIM];

  for (int i = 0; i < DIM*DIM; i++) {
    a[i] = rand() % 1000;
    b[i] = rand() % 1000;
  }
  print_mat(a);
  print_mat(b);

  int * c1 = new int[DIM*DIM];
  clock_t t1 = get_time();
  mat_mul_aligned(a, b, c1);
  t1 = get_time() - t1;
  print_mat(c1);
  //cout << "Aligned T1: " << t1 << endl;
  cout << "Aligned MFlops: " << report_mflops((double)t1/1000) << " Time: " << t1 << endl;

  int * c3 = new int[DIM*DIM];
  clock_t t3 = get_time();
  mat_mul_parallel(a, b, c3);
  t3 = get_time() - t3;
  print_mat(c3);
  //cout << "Parallel T2: " << t3 << endl;
  cout << "Parallel MFlops: " << report_mflops((double)t3/1000) << " Time: " << t3 << endl;
  check(c1, c3);

  int * c4 = new int[DIM*DIM];
  clock_t t4 = get_time();
  mat_mul_block(a, b, c4);
  t4 = get_time() - t4;
  print_mat(c4);
  //cout << "Blocked T2: " << t4 << endl;
  cout << "Blocked MFlops: " << report_mflops((double)t4/1000) << " Time: " << t4 << endl;
  check(c4, c3);
  

  int * c5 = new int[DIM*DIM];
  clock_t t5 = get_time();
  mat_mul_block_par(a, b, c5);
  t5 = get_time() - t5;
  print_mat(c5);
  //cout << "Parallel Blocked T2: " << t5 << endl;
  cout << "Parallel Blocked MFlops: " << report_mflops((double)t5/1000) << " Time: " << t5 << endl;
  check(c5, c4);
  exit(0);

  int * c6 = new int[DIM*DIM];
  clock_t t6 = get_time();
  mat_mul_block2(a, b, c6);
  t6 = get_time() - t6;
  print_mat(c6);
  cout << "Reordered Blocked MFlops: " << report_mflops((double)t6/1000) << " Time: " << t6 << endl;
  check(c6, c5);
}
