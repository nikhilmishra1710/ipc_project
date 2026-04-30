CC=gcc
MPICC=mpicc
NVCC=nvcc
CFLAGS=-O3 -ffast-math -march=native -Wall
OMPFL=-fopenmp
LIBS=-lm
_CC:=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null|head -1|tr -d '. ')
ifneq ($(_CC),)
  NVARCH:=sm_$(_CC)
else
  NVARCH:=sm_61
endif

.PHONY: all clean help
all: seq_nbody omp_nbody mpi_nbody
	@echo "Built: seq_nbody omp_nbody mpi_nbody"
	@echo "For CUDA: make cuda_nbody"

seq_nbody: seq_nbody.c nbody_utils.c nbody.h
	$(CC) $(CFLAGS) -o $@ seq_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"
omp_nbody: omp_nbody.c nbody_utils.c nbody.h
	$(CC) $(CFLAGS) $(OMPFL) -o $@ omp_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"
mpi_nbody: mpi_nbody.c nbody_utils.c nbody.h
	$(MPICC) $(CFLAGS) -o $@ mpi_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"
nbody_utils.o: nbody_utils.c nbody.h
	$(CC) $(CFLAGS) -c nbody_utils.c -o nbody_utils.o
cuda_nbody: cuda_nbody.cu nbody_utils.o nbody.h
	$(NVCC) -O2 -arch=$(NVARCH) -o $@ cuda_nbody.cu nbody_utils.o $(LIBS)
	@echo "[OK] $@ ($(NVARCH))"
cuda_cpu: cuda_nbody.cu nbody_utils.c nbody.h
	$(CC) $(CFLAGS) -DNO_CUDA -x c -o cuda_nbody cuda_nbody.cu nbody_utils.c $(LIBS)
	@echo "[OK] cuda_nbody (CPU fallback)"
clean:
	rm -f seq_nbody omp_nbody mpi_nbody cuda_nbody *.o ui/results.js
