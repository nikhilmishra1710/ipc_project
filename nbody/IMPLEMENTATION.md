# N-Body Gravitational Simulation — Problem Statement & Implementations

---

## Problem Statement

The **N-Body problem** simulates the motion of `N` particles under mutual gravitational attraction.
Each particle has a 3-D position `(x, y, z)`, velocity `(vx, vy, vz)`, and mass `m`.

At every discrete time step:

1. **Force computation** — For each particle `i`, sum the gravitational forces exerted by every other particle `j`:

   ```
   F_ij = G * m_i * m_j / (|r_ij|² + ε²)
   ```

   where `ε` (softening length) prevents the singularity when two particles are very close.

2. **Velocity update** (Euler integration):

   ```
   v_i  +=  (F_i / m_i) * Δt
   ```

3. **Position update**:

   ```
   r_i  +=  v_i * Δt
   ```

### Parameters used in this project

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `N` | 10 000 – 50 000 | Number of particles |
| `TIMESTEPS` | 10 | Simulation steps |
| `DT` | 0.01 | Time step (seconds) |
| `G` | 6.674 × 10⁻¹¹ | Gravitational constant (SI) |
| `EPS2` | 0.01 | Softening factor squared |
| `TILE_SIZE` | 256 | CUDA shared-memory tile width |

### Complexity

The naïve all-pairs force computation is **O(N²)** per time step, making it the dominant cost and an ideal candidate for parallelisation.

---

## Data Structures

All particle data is stored in **Structure-of-Arrays (SoA)** layout — separate flat arrays for each quantity (`x[]`, `y[]`, `z[]`, `vx[]`, `vy[]`, `vz[]`, `m[]`). SoA enables contiguous memory access patterns which are critical for SIMD vectorisation and GPU coalesced reads.

Particles are initialised once with random positions in a 200 m cube, random velocities in [−1, 1] m/s, and masses in [1, 10] kg (fixed seed = 42 for reproducibility).

---

## 1. Sequential Implementation (`seq_nbody.c`)

### Algorithm

```
FUNCTION main(N):
    Allocate arrays: x, y, z, vx, vy, vz, m  [size N]
    Initialise particles with random state (seed=42)

    FOR t = 1 to TIMESTEPS:
        CALL step(x, y, z, vx, vy, vz, m, N)

    Print elapsed time and position checksum
    Free memory

FUNCTION step(x, y, z, vx, vy, vz, m, N):

    // --- Phase 1: Force accumulation + velocity update ---
    FOR each particle i in [0, N):
        fx = fy = fz = 0
        xi = x[i], yi = y[i], zi = z[i]

        FOR each particle j in [0, N):
            dx = x[j] - xi
            dy = y[j] - yi
            dz = z[j] - zi
            dist_sq = dx² + dy² + dz² + EPS2
            inv     = 1 / sqrt(dist_sq)          // 1/r
            f       = G * m[j] * inv³             // G*m_j / r²  (per unit m_i)
            fx += f * dx
            fy += f * dy
            fz += f * dz

        vx[i] += DT * fx
        vy[i] += DT * fy
        vz[i] += DT * fz

    // --- Phase 2: Position update (must follow full velocity update) ---
    FOR each particle i in [0, N):
        x[i] += DT * vx[i]
        y[i] += DT * vy[i]
        z[i] += DT * vz[i]
```

### Key Details
- `__restrict__` keyword on all array pointers tells the compiler the arrays do not alias, enabling **auto-vectorisation** (AVX/SSE).
- The two-phase loop (velocity first, then position) ensures all forces are computed with positions from the *same* timestep before any position is moved.
- **Complexity:** O(N²) per timestep.

---

## 2. OpenMP Implementation (`omp_nbody.c`)

### Algorithm

```
FUNCTION main(N):
    Allocate and initialise arrays (same as Sequential)
    Query number of OpenMP threads

    // Fork a team of threads — they persist across all timesteps
    PARALLEL REGION:
        FOR t = 1 to TIMESTEPS:

            // Phase 1: Divide particles across threads (static schedule)
            PARALLEL FOR i in [0, N):
                Compute net force on particle i from all j in [0, N)
                (same inner loop as Sequential)
                vx[i] += DT * fx
                vy[i] += DT * fy
                vz[i] += DT * fz
            // Implicit barrier here — all velocities updated before positions

            // Phase 2: Parallel position update
            PARALLEL FOR i in [0, N):
                x[i] += DT * vx[i]
                y[i] += DT * vy[i]
                z[i] += DT * vz[i]
            // Implicit barrier — positions consistent for next timestep

    Print elapsed time and checksum
    Free memory
```

### Key Details
- A **single persistent parallel region** wraps both phases and the time loop to avoid repeated fork/join overhead.
- `#pragma omp for schedule(static)` gives each thread a contiguous block of `N / num_threads` particles.
- The implicit barrier at the end of each `omp for` prevents any thread from starting Phase 2 before all forces are computed.
- Each thread reads *all* `N` positions but writes only its own slice of velocities — no race conditions.
- **Complexity:** O(N² / T) per timestep where T = thread count.

---

## 3. MPI Implementation (`mpi_nbody.c`)

### Algorithm

```
FUNCTION main(N):
    MPI_Init — start P processes (ranks 0..P-1)
    rank = this process's ID
    size = total number of processes

    n = N trimmed to exact multiple of P
    chunk = n / P
    lo = rank * chunk
    hi = lo + chunk               // This rank owns particles [lo, hi)

    Allocate full-size arrays: x, y, z, vx, vy, vz, m  [size n]
    Allocate interleaved buffer buf [size n×6]

    IF rank == 0:
        Initialise all particles (only root has initial data)

    // Distribute initial state to all ranks
    MPI_Bcast x, y, z, vx, vy, vz, m from rank 0

    MPI_Barrier — synchronise start time
    t0 = MPI_Wtime()

    FOR t = 1 to TIMESTEPS:

        // Phase 1: Each rank computes forces only for its slice [lo, hi)
        //          but reads ALL N positions (full data is on every rank)
        FOR i in [lo, hi):
            fx = fy = fz = 0
            FOR j in [0, n):                    // all-to-all interaction
                compute force contribution from j
            vx[i] += DT * fx
            vy[i] += DT * fy
            vz[i] += DT * fz

        // Phase 2: Position update for this rank's slice
        FOR i in [lo, hi):
            x[i] += DT * vx[i]
            y[i] += DT * vy[i]
            z[i] += DT * vz[i]

        // Phase 3: Global synchronisation — pack 6 floats per particle
        FOR i in [lo, hi):
            buf[i*6 + 0..5] = x[i], y[i], z[i], vx[i], vy[i], vz[i]

        MPI_Allgather(in-place, chunk*6 floats per rank)
                      // Every rank now has updated positions+velocities
                      // for ALL particles

        // Unpack buffer back to arrays
        FOR i in [0, n):
            x[i], y[i], z[i]   = buf[i*6 + 0..2]
            vx[i], vy[i], vz[i] = buf[i*6 + 3..5]

    MPI_Barrier
    IF rank == 0: print time and checksum
    MPI_Finalize
```

### Key Details
- **Domain decomposition:** each rank is responsible for a contiguous slice of particles. The inner `j` loop still iterates all `N` particles so pairwise forces are exact.
- **Interleaved Allgather:** both positions *and* velocities are packed into one buffer and exchanged in a single `MPI_Allgather`. Synchronising only positions (common bug) leaves stale velocities on other ranks, producing wrong forces in the next step.
- Communication cost per timestep: `6 × N × 4 bytes` of data exchanged across all ranks.
- **Complexity:** O(N² / P) computation + O(N) communication per timestep.

---

## 4. CUDA Implementation (`cuda_nbody.cu`)

### Algorithm

```
FUNCTION main(N):
    Allocate host arrays: x, y, z, vx, vy, vz, m  [size N]
    Initialise particles (CPU)
    Copy all arrays to GPU device memory

    blocks = ceil(N / TILE_SIZE)
    threads_per_block = TILE_SIZE  (256)

    FOR t = 1 to TIMESTEPS:
        LAUNCH forces_kernel<<<blocks, TILE_SIZE>>>
        LAUNCH update_kernel<<<blocks, TILE_SIZE>>>

    Copy result positions from GPU back to CPU
    Print elapsed time and checksum
    Free host and device memory

GPU KERNEL: forces_kernel(x, y, z, vx, vy, vz, mass, N, dt)
    Shared memory arrays: sx[TILE_SIZE], sy[TILE_SIZE], sz[TILE_SIZE], sm[TILE_SIZE]
    i = global thread index

    Load my particle's position: px = x[i], py = y[i], pz = z[i]
    fx = fy = fz = 0

    FOR tile_start = 0 to N step TILE_SIZE:

        // Cooperative load: each thread loads one j-particle into shared mem
        j = tile_start + threadIdx.x
        sx[threadIdx.x] = x[j]
        sy[threadIdx.x] = y[j]
        sz[threadIdx.x] = z[j]
        sm[threadIdx.x] = mass[j]
        __syncthreads()           // wait for all threads to finish loading

        // Each thread computes interaction with all TILE_SIZE loaded particles
        FOR k = 0 to TILE_SIZE:
            dx = sx[k] - px
            dy = sy[k] - py
            dz = sz[k] - pz
            dist_sq = dx² + dy² + dz² + EPS2
            inv = rsqrtf(dist_sq)             // GPU fast reciprocal sqrt
            f   = sm[k] * inv³
            fx += f * dx; fy += f * dy; fz += f * dz

        __syncthreads()           // wait before next tile load

    IF i < N:
        vx[i] += dt * fx
        vy[i] += dt * fy
        vz[i] += dt * fz

GPU KERNEL: update_kernel(x, y, z, vx, vy, vz, N, dt)
    i = global thread index
    IF i < N:
        x[i] += dt * vx[i]
        y[i] += dt * vy[i]
        z[i] += dt * vz[i]
```

### Key Details
- **Tiled shared memory:** the inner loop is broken into tiles of `TILE_SIZE = 256` particles. Each tile is cooperatively loaded into fast shared memory (L1-level) before the force calculation, dramatically reducing global memory bandwidth.
- Each GPU thread handles exactly **one particle** (`i`). The outer tile loop iterates over all `N` source particles in chunks.
- `rsqrtf()` is a hardware-accelerated single-precision reciprocal square root — much faster than `1.0f / sqrtf()` on GPU.
- Two `__syncthreads()` per tile: one after loading (ensure all data is ready) and one after computing (prevent early overwrite of shared data).
- **Complexity:** O(N² / (blocks × threads)) per kernel launch, but with high memory throughput due to shared-memory reuse.

---

## 5. Hybrid MPI + OpenMP Implementation (`hybrid_nbody.c`)

### Algorithm

```
FUNCTION main(N):
    MPI_Init_thread(MPI_THREAD_FUNNELED)
        // Guarantees only the master thread calls MPI routines

    rank = this process's ID
    size = total number of MPI ranks

    n = N trimmed to multiple of size
    chunk = n / size
    lo = rank * chunk
    hi = lo + chunk

    Allocate full arrays + interleaved buffer buf[n×6]

    IF rank == 0: initialise all particles
    MPI_Bcast all arrays from rank 0

    Query OpenMP thread count (before timing)

    MPI_Barrier; t0 = MPI_Wtime()

    FOR t = 1 to TIMESTEPS:

        // Phase 1: OpenMP parallelises force loop within this rank's slice
        PARALLEL FOR (OpenMP) i in [lo, hi):
            fx = fy = fz = 0
            FOR j in [0, n):             // all N particles as sources
                compute G * m[j] / r² force
            vx[i] += DT * fx
            vy[i] += DT * fy
            vz[i] += DT * fz
        // OpenMP barrier (implicit at end of parallel for)

        // Phase 2: OpenMP parallelises position update
        PARALLEL FOR (OpenMP) i in [lo, hi):
            x[i] += DT * vx[i]
            y[i] += DT * vy[i]
            z[i] += DT * vz[i]
        // OpenMP barrier

        // Phase 3: Master thread packs and exchanges with MPI
        FOR i in [lo, hi):
            buf[i*6 + 0..5] = x, y, z, vx, vy, vz

        MPI_Allgather(in-place, chunk*6 floats)

        FOR i in [0, n):
            Unpack buf → x, y, z, vx, vy, vz

    MPI_Barrier
    IF rank == 0: print time and checksum
    MPI_Finalize
```

### Key Details
- **Two-level parallelism:** MPI decomposes the particle domain across nodes/processes; OpenMP parallelises the compute-intensive inner loops within each process using all available CPU cores.
- `MPI_THREAD_FUNNELED` level is requested so that only the master (thread 0) invokes MPI calls — OpenMP threads never touch MPI.
- This mirrors a real HPC cluster setup: one MPI rank per node, multiple OpenMP threads per rank exploiting shared memory within a node.
- **Complexity:** O(N² / (P × T)) per timestep where P = MPI ranks and T = OpenMP threads per rank.

---

## Summary Table

| Implementation | Parallelism Model | Key Mechanism | Complexity / Step |
|---|---|---|---|
| Sequential | None | `__restrict__` auto-vec | O(N²) |
| OpenMP | Shared memory | `#pragma omp parallel for` | O(N²/T) |
| MPI | Distributed memory | Domain split + `Allgather` | O(N²/P) |
| CUDA | GPU massively parallel | Tiled shared-memory kernel | O(N²/GPU threads) |
| Hybrid MPI+OMP | Distributed + shared | MPI ranks × OMP threads | O(N²/(P×T)) |
