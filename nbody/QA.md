# N-Body Simulation — Questions & Answers

---

## Part 1 — Problem & General Design

---

**Q1. What is the N-Body problem and why is it computationally expensive?**

The N-Body problem computes the pairwise gravitational forces among N particles and integrates their trajectories forward in time. The naïve all-pairs approach requires computing N × (N − 1) interactions per timestep, giving **O(N²)** complexity per step. For N = 50 000 that is 2.5 × 10⁹ floating-point operations per step — clearly expensive and motivating parallelisation.

---

**Q2. Why are positions and velocities stored in Structure-of-Arrays (SoA) instead of Array-of-Structures (AoS)?**

In AoS, data for one particle occupies one struct: `{x, y, z, vx, vy, vz, m}`. When iterating over all j-particles to compute forces on particle i, only `x[j]`, `y[j]`, `z[j]`, and `m[j]` are needed — in AoS, the unused velocity fields pollute the cache line.

In SoA, `x[]`, `y[]`, `z[]` are contiguous arrays. The inner j-loop streams through exactly the memory it needs, maximising **cache-line utilisation** and enabling the compiler to emit SIMD (AVX/SSE) gather instructions automatically.

---

**Q3. What is the softening factor `EPS2` and why is it needed?**

When two particles are very close, `|r_ij| → 0` and the Newtonian force `G m_i m_j / r²` would diverge to infinity, making numerical integration impossible. The softening parameter `ε` replaces the denominator with `r² + ε²`, bounding the maximum force to a finite value. This is a standard technique in numerical N-body simulations and introduces only negligible error for well-separated particles.

---

**Q4. What does `__restrict__` do and why is it applied to pointer parameters?**

`__restrict__` is a C99 qualifier that promises the compiler that two pointers do not alias (point to overlapping memory regions). Without it, the compiler must assume that writing to `vx[i]` could change `x[j]`, preventing vectorisation of the inner loop. With `__restrict__`, the compiler is free to load array elements into SIMD registers, unroll loops, and emit AVX/SSE instructions, often yielding a 2–4× speedup from vectorisation alone.

---

**Q5. Why is the force computation split from the position update into two separate loops?**

If positions were updated immediately after each particle's velocity was changed, particle `i+1` would compute its force using the _already-moved_ position of particle `i`, breaking physical consistency. The correct approach is the **kick-drift** (Euler) scheme:

1. Compute forces for all particles using positions at time `t`.
2. Update all velocities.
3. Update all positions using the new velocities.

Mixing these phases introduces a first-order time-integration error proportional to Δt.

---

**Q6. How is correctness verified across different implementations?**

A **checksum** is computed as the sum of all x + y + z coordinates after all timesteps. Because every implementation is initialised with the same random seed (42) and applies identical physics equations, all correct implementations must produce the same checksum value. Any deviation indicates a bug (e.g., unsynchronised state, race condition, missed MPI exchange).

---

## Part 2 — OpenMP

---

**Q7. What does `#pragma omp parallel for schedule(static)` do?**

- `#pragma omp parallel` forks a team of threads.
- `for` distributes loop iterations among threads.
- `schedule(static)` divides the `N` iterations into equal-sized contiguous blocks assigned to threads at compile time, with no dynamic work stealing. For a balanced load (each iteration takes equal time), static scheduling has minimal overhead.

---

**Q8. Why is the parallel region placed _outside_ the time loop instead of inside?**

Creating and destroying a thread team (fork/join) has overhead of ~1–10 µs per operation. Placing `#pragma omp parallel` _outside_ the timestep loop creates the team once and reuses it across all `TIMESTEPS` iterations. Each OpenMP `for` directive inside the loop then re-distributes work to the already-running threads — eliminating repeated fork/join cost.

---

**Q9. Why is there an implicit barrier between the velocity and position update phases in OpenMP?**

OpenMP's `#pragma omp for` has an **implicit barrier** at its closing brace. No thread proceeds past it until all threads have finished their chunk. This ensures:

- All velocities are fully updated before any position update begins (correctness).
- No thread reads a position that has already been moved by another thread.

This barrier is the OpenMP equivalent of the two-phase loop structure used in the sequential version.

---

**Q10. What is a data race and could one occur in the OpenMP implementation?**

A data race happens when two threads read and write the same memory location concurrently without synchronisation. In the force loop, thread A reads `x[j]` to compute the force on `x[i]` — reading is safe from multiple threads simultaneously. Thread A writes only `vx[i]`, `vy[i]`, `vz[i]` for its own particles (static schedule gives disjoint index sets). Therefore **no data race exists** in this implementation.

---

**Q11. What is the difference between `omp_get_wtime()` and `clock()`?**

- `clock()` measures **CPU time** consumed by the calling process — it sums time across all threads, so with 8 threads it may report 8× the wall time.
- `omp_get_wtime()` measures **wall-clock time** (elapsed real time). For benchmarking parallel programs, wall time is the correct metric because it reflects actual user-facing latency.

---

## Part 3 — MPI

---

**Q12. What is `MPI_Bcast` and when is it used here?**

`MPI_Bcast` sends data from one root process to all other processes in a communicator. It is used at startup (after rank 0 initialises the particles) to distribute the initial positions, velocities, and masses to every rank. Without this, only rank 0 would have valid data and all other ranks would compute forces on uninitialised memory.

---

**Q13. What is `MPI_Allgather` and why is it preferred over `MPI_Gather` + `MPI_Bcast`?**

- `MPI_Gather` collects data from all ranks to one root.
- `MPI_Bcast` redistributes from root to all.
- `MPI_Allgather` combines both in a single collective: each rank contributes a chunk, and all ranks receive the complete assembled result. It is typically implemented with an optimised algorithm (e.g., recursive doubling) that requires O(log P) steps instead of two sequential steps.

---

**Q14. Why must _both_ positions and velocities be synchronised via `MPI_Allgather`, not just positions?**

Consider two ranks A and B, each owning half the particles. After timestep 1:

- Rank A has updated `vx[i]` for its particles.
- Rank B still has the _old_ `vx[i]` for those particles.

If only positions are broadcast, in timestep 2 rank B computes force on its particles using correct positions but then when B later packs its results, it uses stale velocities for the particles it doesn't own — producing incorrect future trajectories. Both positions and velocities must be exchanged every step.

---

**Q15. What does `MPI_IN_PLACE` mean in the `MPI_Allgather` call?**

`MPI_IN_PLACE` as the send buffer tells MPI that each rank's contribution is already in the correct location within the receive buffer. This avoids an extra copy and allows MPI to use the receive buffer itself as both input and output — reducing memory bandwidth and often improving performance.

---

**Q16. What is `MPI_Barrier` and why is it called before timing starts?**

`MPI_Barrier` blocks until **all** processes in the communicator reach the call. Without it, a fast rank might start the timestep loop before slower ranks finish broadcasting data, and `MPI_Wtime()` would capture an unequal starting point, skewing the measured time. The barrier ensures all ranks begin the simulation simultaneously.

---

**Q17. How does domain decomposition scale with the number of MPI ranks?**

Each rank processes `N / P` particles in the force loop (O(N²/P) work). Communication is `O(N)` per step (the Allgather sends 6 × N floats total). For large N, the computation dominates and near-linear speedup is expected. At small N, the `O(N)` communication cost and MPI library overhead become non-negligible, and speedup saturates or reverses.

---

## Part 4 — CUDA

---

**Q18. What is a CUDA kernel, block, and thread?**

- A **kernel** is a function that runs on the GPU, declared with `__global__`.
- A **block** is a group of threads that share fast shared memory and can synchronise via `__syncthreads()`.
- A **thread** is the finest unit of execution. The GPU schedules groups of 32 threads (a **warp**) together in SIMD fashion.

In this project: `blocks = ceil(N / 256)`, `threads per block = 256`. Each thread handles one particle.

---

**Q19. What is shared memory on a GPU and why is it used in the tiled kernel?**

Shared memory is a small, fast, on-chip memory (~48 KB per block) with ~100× lower latency than global (device) DRAM. Without tiling, computing the force on particle `i` from all `N` source particles would issue `N` global memory reads — every read potentially a cache miss.

With tiling, the block cooperatively loads `TILE_SIZE = 256` j-particles into shared memory in one coalesced read. All threads in the block then compute against those 256 particles from fast shared memory. This reduces global memory traffic by a factor of `TILE_SIZE`, dramatically increasing arithmetic intensity (FLOPs per byte).

---

**Q20. What is `__syncthreads()` and why are two barriers needed per tile?**

`__syncthreads()` is a block-level synchronisation barrier — all threads in the block must reach it before any thread proceeds.

- **First barrier (after loading):** ensures all 256 j-particles are in shared memory before any thread starts using them. Without it, thread 0 might start computing with data that thread 255 has not yet loaded.
- **Second barrier (after computing):** ensures all threads have finished reading the shared data before the next tile overwrites it. Without it, a fast thread might load tile `k+1` while another thread is still reading tile `k`.

---

**Q21. What is `rsqrtf()` and why is it faster on GPU than `1.0f / sqrtf()`?**

`rsqrtf(x)` computes `1 / sqrt(x)` as a **single GPU hardware instruction** using a Newton-Raphson approximation with ~23-bit precision. On NVIDIA GPUs it executes in 2–4 clock cycles. `1.0f / sqrtf(x)` requires two instructions (sqrt + reciprocal divide), each taking similar cycles. Using `rsqrtf` therefore reduces instruction count and latency, which matters at the scale of billions of evaluations per second.

---

**Q22. What is memory coalescing and does this kernel achieve it?**

Memory coalescing occurs when consecutive threads in a warp access consecutive memory addresses, allowing the GPU to issue a single wide memory transaction instead of multiple individual ones.

In `forces_kernel`, the tile loading step `sx[threadIdx.x] = x[tile_start + threadIdx.x]` accesses `x[]` with consecutive indices for consecutive threads — this is **perfectly coalesced**. The velocity update `vx[i] += ...` writes to `vx[blockIdx.x * TILE_SIZE + threadIdx.x]` — also coalesced. So yes, the kernel is designed for coalesced access.

---

**Q23. What is the difference between `cudaMemcpy` directions `HostToDevice` and `DeviceToHost`?**

- `cudaMemcpyHostToDevice` (H2D): copies data from CPU RAM to GPU VRAM. Used at startup to transfer initial particle state.
- `cudaMemcpyDeviceToHost` (D2H): copies data from GPU VRAM back to CPU RAM. Used after all timesteps to retrieve the final positions for checksum computation.

Only positions are copied back (not velocities) because the checksum only requires `x`, `y`, `z`.

---

**Q24. Why is `cudaDeviceSynchronize()` called after each kernel launch?**

CUDA kernel launches are **asynchronous** — the CPU does not wait for the GPU to finish before continuing. Without `cudaDeviceSynchronize()`:

- The `forces_kernel` might not be complete when `update_kernel` starts, reading partially updated velocities.
- Timing measured on the CPU would not include the actual GPU compute time.

`cudaDeviceSynchronize()` blocks the CPU until all previously launched GPU work completes, ensuring correct ordering and accurate timing.

---

## Part 5 — Hybrid MPI + OpenMP

---

**Q25. What does `MPI_THREAD_FUNNELED` mean?**

It is a thread-safety level requested via `MPI_Init_thread`. `MPI_THREAD_FUNNELED` guarantees that the MPI library supports being called from a multi-threaded process, but **only the master thread** (the thread that called `MPI_Init_thread`) may call MPI functions. OpenMP worker threads must not call MPI. This is the minimum level needed for a fork-join model where MPI calls happen outside OpenMP parallel regions.

---

**Q26. How does the hybrid model combine MPI and OpenMP?**

- **MPI (inter-process):** divides the N particles across P independent processes (ranks). Each rank owns a slice `[lo, hi)`. MPI handles all data exchange between processes.
- **OpenMP (intra-process):** within each rank, the force and position loops over the local slice are parallelised across T threads sharing the same RAM.

Total effective parallelism = **P × T**. On a cluster with 4 nodes × 8 cores each, P = 4 and T = 8 gives 32-way parallelism.

---

**Q27. Why is it important that MPI calls are made only by the master thread in the hybrid model?**

OpenMP threads within the same rank all share the same MPI state (rank, communicator, request objects). If multiple threads called `MPI_Allgather` simultaneously on the same communicator, the internal MPI message-matching and buffer management would have undefined behaviour under `MPI_THREAD_FUNNELED`. Using `MPI_THREAD_MULTIPLE` would allow it but adds significant locking overhead inside the MPI library.

---

**Q28. What is the Amdahl's Law implication for the hybrid implementation?**

Amdahl's Law states that speedup is limited by the serial fraction `s`:

```
Speedup ≤ 1 / (s + (1-s)/P)
```

In this simulation:

- The MPI `Allgather` is inherently serial in the sense that all ranks must wait for it.
- Communication time scales as O(N) per step and does not shrink with more ranks or threads.

As P and T grow, the parallel compute portion shrinks per rank, but the O(N) communication remains constant — eventually communication dominates and additional parallelism yields diminishing returns.

---

**Q29. When should you choose MPI vs OpenMP vs Hybrid?**

| Scenario                                          | Recommendation                                      |
| ------------------------------------------------- | --------------------------------------------------- |
| Single multi-core machine                         | OpenMP — simplest, lowest latency                   |
| Multiple machines (cluster)                       | MPI — the only option across separate memory spaces |
| Cluster where each node has many cores            | Hybrid — MPI between nodes, OpenMP within each node |
| Massive data parallelism, regular access patterns | CUDA GPU — orders of magnitude higher throughput    |
| Prototyping / correctness reference               | Sequential                                          |

---

**Q30. What was the MPI velocity synchronisation bug and how was it fixed?**

**Bug:** The original implementation used `MPI_Allgather` to share only positions (x, y, z). Velocities were not exchanged. After each timestep, each rank had up-to-date positions for all particles, but stale velocities for particles it did not own. In the next timestep, when these particles were moved from the gathered positions, the velocity used was wrong, causing the simulation to diverge from the correct trajectory.

**Fix:** An interleaved buffer of 6 floats per particle `[x, y, z, vx, vy, vz]` is packed for the local slice, and a single `MPI_Allgather` exchanges all 6 values per particle in one collective. After gathering, all ranks unpack both positions and velocities, ensuring a fully consistent global state for every new timestep.
