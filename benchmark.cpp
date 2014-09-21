//Uses the eigen library
#include <iostream>
#include <Eigen/Dense>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>

#define DIM 1024

using namespace Eigen;

long long int get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

double report_mflops(double t) {
  double n = DIM;
  double f = (2.0*(n*n*n) / t) / 1000000;
  return f;
}

int main(int argc, char *argv[]) {
  
  srand(time(NULL));
  MatrixXd mat1(DIM, DIM), mat2(DIM, DIM);
  
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      mat1(i, j) = rand() % 100;
      mat2(i, j) = rand() % 100;
    }
  }
  
  MatrixXd mat3(DIM, DIM);
  clock_t start = get_time();
  mat3 = mat1 * mat2;
  clock_t elapsed = get_time() - start;
  std::cout << "MFlops: " << report_mflops(elapsed/1000.0f) << " Time: " << elapsed << std::endl;
}
