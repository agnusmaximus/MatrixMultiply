all:
		g++ -fopenmp -O3 -msse4 benchmark.cpp -I/opt/local/include/eigen2/ -o benchmark
		g++ -O3 -fopenmp -mavx2 mat_mul.cpp -o mat_mul
		g++ -march=native -O3 -fopenmp -mavx2 mat_mul_float.cpp -o mat_mul_float
