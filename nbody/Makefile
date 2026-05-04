CC=gcc
MPICC=mpicc
NVCC=nvcc
CFLAGS=-O3 -ffast-math -march=native -Wall
OMPFL=-fopenmp
LIBS=-lm
BUILD_DIR=build
_CC:=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null|head -1|tr -d '. ')
ifneq ($(_CC),)
  NVARCH:=sm_$(_CC)
else
  NVARCH:=sm_61
endif

# Sentinel to create build dir once
$(BUILD_DIR)/.dir:
	mkdir -p $(BUILD_DIR)
	@touch $@

.PHONY: all clean help
all: $(BUILD_DIR)/seq_nbody $(BUILD_DIR)/omp_nbody $(BUILD_DIR)/mpi_nbody $(BUILD_DIR)/hybrid_nbody
	@chmod +x run_benchmark.sh
	@echo "Built: $(BUILD_DIR)/seq_nbody $(BUILD_DIR)/omp_nbody $(BUILD_DIR)/mpi_nbody $(BUILD_DIR)/hybrid_nbody"
	@echo "For CUDA: make $(BUILD_DIR)/cuda_nbody"

$(BUILD_DIR)/seq_nbody: seq_nbody.c nbody_utils.c nbody.h | $(BUILD_DIR)/.dir
	$(CC) $(CFLAGS) -o $@ seq_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"

$(BUILD_DIR)/omp_nbody: omp_nbody.c nbody_utils.c nbody.h | $(BUILD_DIR)/.dir
	$(CC) $(CFLAGS) $(OMPFL) -o $@ omp_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"

$(BUILD_DIR)/mpi_nbody: mpi_nbody.c nbody_utils.c nbody.h | $(BUILD_DIR)/.dir
	$(MPICC) $(CFLAGS) -o $@ mpi_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"

$(BUILD_DIR)/hybrid_nbody: hybrid_nbody.c nbody_utils.c nbody.h | $(BUILD_DIR)/.dir
	$(MPICC) $(CFLAGS) $(OMPFL) -o $@ hybrid_nbody.c nbody_utils.c $(LIBS)
	@echo "[OK] $@"

$(BUILD_DIR)/nbody_utils.o: nbody_utils.c nbody.h | $(BUILD_DIR)/.dir
	$(CC) $(CFLAGS) -c nbody_utils.c -o $@

$(BUILD_DIR)/cuda_nbody: cuda_nbody.cu $(BUILD_DIR)/nbody_utils.o nbody.h | $(BUILD_DIR)/.dir
	$(NVCC) -O2 -arch=$(NVARCH) -o $@ cuda_nbody.cu $(BUILD_DIR)/nbody_utils.o $(LIBS)
	@echo "[OK] $@ ($(NVARCH))"

cuda_cpu: cuda_nbody.cu nbody_utils.c nbody.h | $(BUILD_DIR)/.dir
	$(CC) $(CFLAGS) -DNO_CUDA -x c -o $(BUILD_DIR)/cuda_nbody cuda_nbody.cu nbody_utils.c $(LIBS)
	@echo "[OK] $(BUILD_DIR)/cuda_nbody (CPU fallback)"

clean:
	rm -rf $(BUILD_DIR) ui/results.js
