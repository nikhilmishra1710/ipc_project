#!/usr/bin/env bash
set -euo pipefail
BLD='\033[1m';GRN='\033[0;32m';CYN='\033[0;36m';RST='\033[0m'
header(){ echo -e "\n${BLD}${CYN}=== $1 ===${RST}"; }
gt(){ printf '%s' "$1"|tr -d '\r'|grep "^$2:"|awk '{print $2}'|tail -1; }
spd(){ python3 -c "r=float('$1');t=float('$2');print(f'{r/t:.2f}' if t>0 else '0')" 2>/dev/null||echo 0; }
eff(){ python3 -c "t0=float('$1');t=float('$2');print(f'{t0/t:.2f}' if t>0 else '0')" 2>/dev/null||echo 0; }

header "Building"
make build/seq_nbody build/omp_nbody build/mpi_nbody build/hybrid_nbody 2>&1 | tail -4
if command -v nvcc &>/dev/null; then make build/cuda_nbody 2>&1|tail -1; else make cuda_cpu 2>&1|tail -1; fi
echo -e "${GRN}All built.${RST}"

JS_ALL=""

for N in 10000 30000 50000; do
    header "N=$N"
    
    echo -n "  Sequential...      "; O=$(./build/seq_nbody $N 2>&1)
    T_SEQ=$(gt "$O" TIME); echo "${T_SEQ}s"
    
    echo -n "  OpenMP 2T...       "; O=$(OMP_NUM_THREADS=2 ./build/omp_nbody $N 2>&1)
    T_OMP2=$(gt "$O" TIME); echo "${T_OMP2}s"
    
    echo -n "  OpenMP 4T...       "; O=$(OMP_NUM_THREADS=4 ./build/omp_nbody $N 2>&1)
    T_OMP4=$(gt "$O" TIME); echo "${T_OMP4}s"
    
    echo -n "  OpenMP 8T...       "; O=$(OMP_NUM_THREADS=8 ./build/omp_nbody $N 2>&1)
    T_OMP8=$(gt "$O" TIME); echo "${T_OMP8}s"
    
    echo -n "  MPI 2R...          "; O=$(mpirun --allow-run-as-root --oversubscribe -n 2 ./build/mpi_nbody $N 2>&1)
    T_MPI2=$(gt "$O" TIME); echo "${T_MPI2}s"
    
    echo -n "  MPI 4R...          "; O=$(mpirun --allow-run-as-root --oversubscribe -n 4 ./build/mpi_nbody $N 2>&1)
    T_MPI4=$(gt "$O" TIME); echo "${T_MPI4}s"
    
    echo -n "  MPI 8R...          "; O=$(mpirun --allow-run-as-root --oversubscribe -n 8 ./build/mpi_nbody $N 2>&1)
    T_MPI8=$(gt "$O" TIME); echo "${T_MPI8}s"
    
    echo -n "  Hybrid 2R X 2T...  "; O=$(OMP_NUM_THREADS=2 mpirun --allow-run-as-root --oversubscribe -n 2 ./build/hybrid_nbody $N 2>&1)
    T_HYB22=$(gt "$O" TIME); echo "${T_HYB22}s"
    
    echo -n "  Hybrid 2R X 4T...  "; O=$(OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -n 2 ./build/hybrid_nbody $N 2>&1)
    T_HYB24=$(gt "$O" TIME); echo "${T_HYB24}s"
    
    echo -n "  Hybrid 4R X 2T...  "; O=$(OMP_NUM_THREADS=2 mpirun --allow-run-as-root --oversubscribe -n 4 ./build/hybrid_nbody $N 2>&1)
    T_HYB42=$(gt "$O" TIME); echo "${T_HYB42}s"
    
    echo -n "  CUDA...            "; O=$(./build/cuda_nbody $N 2>&1)
    T_CUDA=$(gt "$O" TIME); echo "${T_CUDA}s"
    
    S_OMP2=$(spd "$T_SEQ" "$T_OMP2");   S_OMP4=$(spd "$T_SEQ" "$T_OMP4");   S_OMP8=$(spd "$T_SEQ" "$T_OMP8")
    S_MPI2=$(spd "$T_SEQ" "$T_MPI2");   S_MPI4=$(spd "$T_SEQ" "$T_MPI4");   S_MPI8=$(spd "$T_SEQ" "$T_MPI8")
    S_HYB22=$(spd "$T_SEQ" "$T_HYB22"); S_HYB24=$(spd "$T_SEQ" "$T_HYB24"); S_HYB42=$(spd "$T_SEQ" "$T_HYB42")
    S_CUDA=$(spd "$T_SEQ" "$T_CUDA")
    
    echo ""
    echo -e "${BLD}  N=$N Results:${RST}"
    printf "  %-18s  %-10s  %-8s\n" "Method" "Time(s)" "Speedup"
    printf "  %-18s  %-10s  %-8s\n" "---------------" "--------" "-------"
    printf "  %-18s  %-10s  %-8s\n" "Sequential"     "$T_SEQ"   "1.00x"
    printf "  %-18s  %-10s  %-8s\n" "OpenMP 2T"      "$T_OMP2"  "${S_OMP2}x"
    printf "  %-18s  %-10s  %-8s\n" "OpenMP 4T"      "$T_OMP4"  "${S_OMP4}x"
    printf "  %-18s  %-10s  %-8s\n" "OpenMP 8T"      "$T_OMP8"  "${S_OMP8}x"
    printf "  %-18s  %-10s  %-8s\n" "MPI 2R"         "$T_MPI2"  "${S_MPI2}x"
    printf "  %-18s  %-10s  %-8s\n" "MPI 4R"         "$T_MPI4"  "${S_MPI4}x"
    printf "  %-18s  %-10s  %-8s\n" "MPI 8R"         "$T_MPI8"  "${S_MPI8}x"
    printf "  %-18s  %-10s  %-8s\n" "Hybrid 2R X 2T" "$T_HYB22" "${S_HYB22}x"
    printf "  %-18s  %-10s  %-8s\n" "Hybrid 2R X 4T" "$T_HYB24" "${S_HYB24}x"
    printf "  %-18s  %-10s  %-8s\n" "Hybrid 4R X 2T" "$T_HYB42" "${S_HYB42}x"
    printf "  %-18s  %-10s  %-8s\n" "CUDA"           "$T_CUDA"  "${S_CUDA}x"
    
    JS_ALL="${JS_ALL}{n:${N},seq:${T_SEQ},omp2:${T_OMP2},omp4:${T_OMP4},omp8:${T_OMP8},mpi2:${T_MPI2},mpi4:${T_MPI4},mpi8:${T_MPI8},hyb22:${T_HYB22},hyb24:${T_HYB24},hyb42:${T_HYB42},cuda:${T_CUDA},s_omp2:${S_OMP2},s_omp4:${S_OMP4},s_omp8:${S_OMP8},s_mpi2:${S_MPI2},s_mpi4:${S_MPI4},s_mpi8:${S_MPI8},s_hyb22:${S_HYB22},s_hyb24:${S_HYB24},s_hyb42:${S_HYB42},s_cuda:${S_CUDA}},"
done


# ── CUDA Tile Size Experiment (runtime tile — like OMP_NUM_THREADS) ────────────
TILE_JS=""
if command -v nvcc &>/dev/null; then
    header "CUDA Tile Size Experiment  (N=30000)"
    N_TILE=30000
    for tile in 64 128 256 512; do
        echo -n "  tile=$tile... "
        O=$(./build/cuda_nbody $N_TILE $tile 2>&1)
        T_T=$(gt "$O" TIME); echo "${T_T}s"
        TILE_JS="${TILE_JS}{tile:${tile},time:${T_T}},"
    done
else
    echo -e "\n(Skipping CUDA tile experiment — nvcc not found)"
fi

# ── Write JSON files ───────────────────────────────────────────────────────────
cat > ui/results.js << JSEOF
const RESULTS = { configs: [${JS_ALL}] };
JSEOF

python3 -c "
import json, re, sys
s = open('ui/results.js').read()
m = re.search(r'\[(.+)\]', s, re.DOTALL)
if not m: sys.exit(1)
raw = '[' + m.group(1).rstrip(',') + ']'
raw = re.sub(r'(\b[a-z_][a-z0-9_]*\b):', r'\"\\1\":', raw)
data = json.loads(raw)
json.dump(data, open('ui/results.json','w'), indent=2)
print('ui/results.json written')
" 2>/dev/null || echo "(results.json skipped)"

write_simple_json(){
    local raw="$1" dest="$2"
    python3 -c "
import json, re
raw = '$raw'
raw = '[' + raw.rstrip(',') + ']'
raw = re.sub(r'(\b[a-z_][a-z0-9_]*\b):', r'\"\\1\":', raw)
json.dump(json.loads(raw), open('$dest','w'), indent=2)
print('$dest written')
    " 2>/dev/null || echo "($dest skipped)"
}

[ -n "$TILE_JS" ] && write_simple_json "$TILE_JS" "ui/tile_results.json"

echo ""
echo -e "${GRN}Generated JSON files in ui/ — open ui/index.html in browser${RST}"

# ── Generate charts ────────────────────────────────────────────────────────────
if python3 -c "import matplotlib" 2>/dev/null; then
    echo -e "${CYN}Generating charts...${RST}"
    python3 plot_results.py && echo -e "${GRN}Charts saved → ui/${RST}"
else
    echo "(matplotlib not found — skipping charts)"
fi
