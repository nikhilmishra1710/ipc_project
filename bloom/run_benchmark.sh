#!/usr/bin/env bash
set -euo pipefail
BLD='\033[1m';GRN='\033[0;32m';CYN='\033[0;36m';RST='\033[0m'
header(){ echo -e "\n${BLD}${CYN}=== $1 ===${RST}"; }
gt(){ printf '%s' "$1"|tr -d '\r'|grep "^$2:"|awk '{print $2}'|tail -1; }
spd(){ python3 -c "r=float('$1');t=float('$2');print(f'{r/t:.2f}' if t>0 else '0')" 2>/dev/null||echo 0; }

header "Building"
make seq_bloom omp_bloom mpi_bloom 2>&1|tail -3
if command -v nvcc &>/dev/null; then make cuda_bloom 2>&1|tail -1; else make cuda_cpu 2>&1|tail -1; fi
echo -e "${GRN}All built.${RST}"

JS=""
for N in 1000000 10000000 75000000; do
    NK=$((N/1000000))M
    header "N=${NK} URLs"

    echo -n "  Seq... ";     O=$(./seq_bloom $N 2>&1)
    T_SEQ=$(gt "$O" TIME); FP_SEQ=$(gt "$O" FPR_V); FN_SEQ=$(gt "$O" FN_V); B_SEQ=$(gt "$O" BITS_V); echo "${T_SEQ}s"

    echo -n "  OMP 2T... ";  O=$(OMP_NUM_THREADS=2 ./omp_bloom $N 2>&1)
    T_O2=$(gt "$O" TIME); FP_O2=$(gt "$O" FPR_V); FN_O2=$(gt "$O" FN_V); echo "${T_O2}s"

    echo -n "  OMP 4T... ";  O=$(OMP_NUM_THREADS=4 ./omp_bloom $N 2>&1)
    T_O4=$(gt "$O" TIME); FP_O4=$(gt "$O" FPR_V); FN_O4=$(gt "$O" FN_V); echo "${T_O4}s"

    echo -n "  OMP 8T... ";  O=$(OMP_NUM_THREADS=8 ./omp_bloom $N 2>&1)
    T_O8=$(gt "$O" TIME); FP_O8=$(gt "$O" FPR_V); FN_O8=$(gt "$O" FN_V); echo "${T_O8}s"

    echo -n "  MPI 2R... ";  O=$(mpirun --allow-run-as-root --oversubscribe -n 2 ./mpi_bloom $N 2>&1)
    T_M2=$(gt "$O" TIME); FP_M2=$(gt "$O" FPR_V); FN_M2=$(gt "$O" FN_V); echo "${T_M2}s"

    echo -n "  MPI 4R... ";  O=$(mpirun --allow-run-as-root --oversubscribe -n 4 ./mpi_bloom $N 2>&1)
    T_M4=$(gt "$O" TIME); FP_M4=$(gt "$O" FPR_V); FN_M4=$(gt "$O" FN_V); echo "${T_M4}s"

    echo -n "  MPI 8R... ";  O=$(mpirun --allow-run-as-root --oversubscribe -n 8 ./mpi_bloom $N 2>&1)
    T_M8=$(gt "$O" TIME); FP_M8=$(gt "$O" FPR_V); FN_M8=$(gt "$O" FN_V); echo "${T_M8}s"

    echo -n "  CUDA... ";    O=$(./cuda_bloom $N 2>&1)
    T_CUDA=$(gt "$O" TIME); FP_CUDA=$(gt "$O" FPR_V); FN_CUDA=$(gt "$O" FN_V); B_CUDA=$(gt "$O" BITS_V); echo "${T_CUDA}s"

    SO2=$(spd "$T_SEQ" "$T_O2"); SO4=$(spd "$T_SEQ" "$T_O4"); SO8=$(spd "$T_SEQ" "$T_O8")
    SM2=$(spd "$T_SEQ" "$T_M2"); SM4=$(spd "$T_SEQ" "$T_M4"); SM8=$(spd "$T_SEQ" "$T_M8")
    SC=$(spd "$T_SEQ" "$T_CUDA")

    echo ""
    echo -e "${BLD}  N=${NK} Results:${RST}"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "Method" "Time(s)" "Speedup" "FPR" "FN"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "Sequential" "$T_SEQ" "1.00x" "$FP_SEQ" "$FN_SEQ"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "OpenMP 2T" "$T_O2" "${SO2}x" "$FP_O2" "$FN_O2"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "OpenMP 4T" "$T_O4" "${SO4}x" "$FP_O4" "$FN_O4"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "OpenMP 8T" "$T_O8" "${SO8}x" "$FP_O8" "$FN_O8"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "MPI 2R" "$T_M2" "${SM2}x" "$FP_M2" "$FN_M2"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "MPI 4R" "$T_M4" "${SM4}x" "$FP_M4" "$FN_M4"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "MPI 8R" "$T_M8" "${SM8}x" "$FP_M8" "$FN_M8"
    printf "  %-12s %-10s %-8s %-10s %-4s\n" "CUDA" "$T_CUDA" "${SC}x" "$FP_CUDA" "$FN_CUDA"

    JS="${JS}{n:${N},seq:${T_SEQ},o2:${T_O2},o4:${T_O4},o8:${T_O8},m2:${T_M2},m4:${T_M4},m8:${T_M8},cuda:${T_CUDA},so2:${SO2},so4:${SO4},so8:${SO8},sm2:${SM2},sm4:${SM4},sm8:${SM8},sc:${SC},fp_seq:${FP_SEQ},fn_seq:${FN_SEQ},bits_seq:${B_SEQ},fp_o:\"${FP_O2}\",fp_m:\"${FP_M2}\",fp_c:\"${FP_CUDA}\",fn_o:${FN_O2},fn_m:${FN_M2},fn_c:${FN_CUDA}},"
done
cat > ui/results.js << JSEOF
const RESULTS={configs:[${JS}]};
JSEOF
echo -e "\n${GRN}Generated ui/results.js — open ui/index.html${RST}"
