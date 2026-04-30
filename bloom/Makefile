CC=gcc
MPICC=mpicc
NVCC=nvcc
CFLAGS=-O2 -Wall -Wextra
OMPFL=-fopenmp
LIBS=-lm
_CC:=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null|head -1|tr -d '. ')
ifneq ($(_CC),)
  NVARCH:=sm_$(_CC)
else
  NVARCH:=sm_61
endif
.PHONY: all clean
all: seq_bloom omp_bloom mpi_bloom
seq_bloom: seq_bloom.c bloom_utils.c bloom.h
	$(CC) $(CFLAGS) -o $@ seq_bloom.c bloom_utils.c $(LIBS)
	@echo "[OK] $@"
omp_bloom: omp_bloom.c bloom_utils.c bloom.h
	$(CC) $(CFLAGS) $(OMPFL) -o $@ omp_bloom.c bloom_utils.c $(LIBS)
	@echo "[OK] $@"
mpi_bloom: mpi_bloom.c bloom_utils.c bloom.h
	$(MPICC) $(CFLAGS) -o $@ mpi_bloom.c bloom_utils.c $(LIBS)
	@echo "[OK] $@"
bloom_utils.o: bloom_utils.c bloom.h
	$(CC) $(CFLAGS) -c bloom_utils.c -o bloom_utils.o
cuda_bloom: cuda_bloom.cu bloom_utils.o bloom.h
	$(NVCC) -O2 -arch=$(NVARCH) -o $@ cuda_bloom.cu bloom_utils.o $(LIBS)
	@echo "[OK] $@ ($(NVARCH))"
cuda_cpu: cuda_bloom.cu bloom_utils.c bloom.h
	$(CC) $(CFLAGS) -DNO_CUDA -x c -o cuda_bloom cuda_bloom.cu bloom_utils.c $(LIBS)
clean:
	rm -f seq_bloom omp_bloom mpi_bloom cuda_bloom *.o ui/results.js
