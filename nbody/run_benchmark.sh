#!/usr/bin/env bash
set -euo pipefail
BLD='\033[1m';GRN='\033[0;32m';CYN='\033[0;36m';RST='\033[0m'
header(){ echo -e "\n${BLD}${CYN}=== $1 ===${RST}"; }
gt(){ printf '%s' "$1"|tr -d '\r'|grep "^$2:"|awk '{print $2}'|tail -1; }
spd(){ python3 -c "r=float('$1');t=float('$2');print(f'{r/t:.2f}' if t>0 else '0')" 2>/dev/null||echo 0; }

header "Building"
make seq_nbody omp_nbody mpi_nbody 2>&1 | tail -3
if command -v nvcc &>/dev/null; then make cuda_nbody 2>&1|tail -1; else make cuda_cpu 2>&1|tail -1; fi
echo -e "${GRN}All built.${RST}"

JS_ALL=""

for N in 10000 30000 50000; do
    header "N=$N"

    echo -n "  Sequential... "; O=$(./seq_nbody $N 2>&1)
    T_SEQ=$(gt "$O" TIME); echo "${T_SEQ}s"

    echo -n "  OpenMP 2T... "; O=$(OMP_NUM_THREADS=2 ./omp_nbody $N 2>&1)
    T_OMP2=$(gt "$O" TIME); echo "${T_OMP2}s"

    echo -n "  OpenMP 4T... "; O=$(OMP_NUM_THREADS=4 ./omp_nbody $N 2>&1)
    T_OMP4=$(gt "$O" TIME); echo "${T_OMP4}s"

    echo -n "  OpenMP 8T... "; O=$(OMP_NUM_THREADS=8 ./omp_nbody $N 2>&1)
    T_OMP8=$(gt "$O" TIME); echo "${T_OMP8}s"

    echo -n "  MPI 2R... "; O=$(mpirun --allow-run-as-root --oversubscribe -n 2 ./mpi_nbody $N 2>&1)
    T_MPI2=$(gt "$O" TIME); echo "${T_MPI2}s"

    echo -n "  MPI 4R... "; O=$(mpirun --allow-run-as-root --oversubscribe -n 4 ./mpi_nbody $N 2>&1)
    T_MPI4=$(gt "$O" TIME); echo "${T_MPI4}s"

    echo -n "  MPI 8R... "; O=$(mpirun --allow-run-as-root --oversubscribe -n 8 ./mpi_nbody $N 2>&1)
    T_MPI8=$(gt "$O" TIME); echo "${T_MPI8}s"

    echo -n "  CUDA... "; O=$(./cuda_nbody $N 2>&1)
    T_CUDA=$(gt "$O" TIME); echo "${T_CUDA}s"

    S_OMP2=$(spd "$T_SEQ" "$T_OMP2"); S_OMP4=$(spd "$T_SEQ" "$T_OMP4"); S_OMP8=$(spd "$T_SEQ" "$T_OMP8")
    S_MPI2=$(spd "$T_SEQ" "$T_MPI2"); S_MPI4=$(spd "$T_SEQ" "$T_MPI4"); S_MPI8=$(spd "$T_SEQ" "$T_MPI8")
    S_CUDA=$(spd "$T_SEQ" "$T_CUDA")

    echo ""
    echo -e "${BLD}  N=$N Results:${RST}"
    printf "  %-14s  %-10s  %-8s\n" "Method" "Time(s)" "Speedup"
    printf "  %-14s  %-10s  %-8s\n" "-----------" "--------" "-------"
    printf "  %-14s  %-10s  %-8s\n" "Sequential" "$T_SEQ" "1.00x"
    printf "  %-14s  %-10s  %-8s\n" "OpenMP 2T" "$T_OMP2" "${S_OMP2}x"
    printf "  %-14s  %-10s  %-8s\n" "OpenMP 4T" "$T_OMP4" "${S_OMP4}x"
    printf "  %-14s  %-10s  %-8s\n" "OpenMP 8T" "$T_OMP8" "${S_OMP8}x"
    printf "  %-14s  %-10s  %-8s\n" "MPI 2R" "$T_MPI2" "${S_MPI2}x"
    printf "  %-14s  %-10s  %-8s\n" "MPI 4R" "$T_MPI4" "${S_MPI4}x"
    printf "  %-14s  %-10s  %-8s\n" "MPI 8R" "$T_MPI8" "${S_MPI8}x"
    printf "  %-14s  %-10s  %-8s\n" "CUDA" "$T_CUDA" "${S_CUDA}x"

    JS_ALL="${JS_ALL}{n:${N},seq:${T_SEQ},omp2:${T_OMP2},omp4:${T_OMP4},omp8:${T_OMP8},mpi2:${T_MPI2},mpi4:${T_MPI4},mpi8:${T_MPI8},cuda:${T_CUDA},s_omp2:${S_OMP2},s_omp4:${S_OMP4},s_omp8:${S_OMP8},s_mpi2:${S_MPI2},s_mpi4:${S_MPI4},s_mpi8:${S_MPI8},s_cuda:${S_CUDA}},"
done

cat > ui/results.js << JSEOF
const RESULTS = { configs: [${JS_ALL}] };
JSEOF
echo ""
echo -e "${GRN}Generated ui/results.js — open ui/index.html in browser${RST}"
