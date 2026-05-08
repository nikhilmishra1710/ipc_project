/*
 * cuda_nbody.cu — CUDA N-Body, Velocity Verlet integrator
 *
 * Tile size is a RUNTIME parameter (argv[2], default 256), implemented via
 * dynamic shared memory — analogous to OMP_NUM_THREADS for OpenMP.
 *
 * Usage:  ./build/cuda_nbody <N> [tile_size]
 * E.g.:   ./build/cuda_nbody 30000 128
 */
#include "nbody.h"

#ifndef NO_CUDA
#    include <cuda_runtime.h>
#    define CK(c)                                                                                                                                    \
        do {                                                                                                                                         \
            cudaError_t e = (c);                                                                                                                     \
            if (e != cudaSuccess) {                                                                                                                  \
                fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                                                       \
                exit(1);                                                                                                                             \
            }                                                                                                                                        \
        } while (0)

/* -----------------------------------------------------------------------
 * forces_kernel — computes acceleration for each particle i.
 * Uses dynamic shared memory: 4 arrays of tile_sz floats each.
 * ----------------------------------------------------------------------- */
__global__ void forces_kernel(const float* x, const float* y, const float* z, const float* mass, float* ax, float* ay, float* az, int n,
                              int tile_sz) {
    extern __shared__ float smem[];
    float* sx = smem;
    float* sy = smem + tile_sz;
    float* sz_s = smem + 2 * tile_sz;
    float* smass = smem + 3 * tile_sz; /* renamed: avoid any sm clash */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = 0, py = 0, pz = 0, fx = 0, fy = 0, fz = 0;
    if (i < n) {
        px = x[i];
        py = y[i];
        pz = z[i];
    }

    for (int tile = 0; tile < n; tile += tile_sz) {
        int j = tile + threadIdx.x;
        sx[threadIdx.x] = (j < n) ? x[j] : 0.0f;
        sy[threadIdx.x] = (j < n) ? y[j] : 0.0f;
        sz_s[threadIdx.x] = (j < n) ? z[j] : 0.0f;
        smass[threadIdx.x] = (j < n) ? mass[j] : 0.0f;
        __syncthreads();

        if (i < n) {
            int lim = min(tile_sz, n - tile);
            for (int k = 0; k < lim; k++) {
                float dx = sx[k] - px, dy = sy[k] - py, dz = sz_s[k] - pz;
                float d2 = dx * dx + dy * dy + dz * dz + EPS2;
                float inv = rsqrtf(d2);
                float f = G * smass[k] * inv * inv * inv;
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }
        }
        __syncthreads();
    }
    if (i < n) {
        ax[i] = fx;
        ay[i] = fy;
        az[i] = fz;
    }
}

/* Position update (Verlet step 1) */
__global__ void verlet_pos_kernel(float* x, float* y, float* z, const float* vx, const float* vy, const float* vz, const float* ax, const float* ay,
                                  const float* az, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float dt2 = 0.5f * dt * dt;
        x[i] += vx[i] * dt + ax[i] * dt2;
        y[i] += vy[i] * dt + ay[i] * dt2;
        z[i] += vz[i] * dt + az[i] * dt2;
    }
}

/* Velocity update (Verlet step 2) */
__global__ void verlet_vel_kernel(float* vx, float* vy, float* vz, const float* ax_old, const float* ay_old, const float* az_old, const float* ax_new,
                                  const float* ay_new, const float* az_new, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float hdt = 0.5f * dt;
        vx[i] += hdt * (ax_old[i] + ax_new[i]);
        vy[i] += hdt * (ay_old[i] + ay_new[i]);
        vz[i] += hdt * (az_old[i] + az_new[i]);
    }
}

static void run_cuda(float* hx, float* hy, float* hz, float* hvx, float* hvy, float* hvz, float* hm, int n, int tile_sz) {
    size_t sz = n * sizeof(float);
    float *dx, *dy, *dz, *dvx, *dvy, *dvz, *dm;
    float *dax, *day, *daz, *dnax, *dnay, *dnaz;

    CK(cudaMalloc((void**)&dx, sz));
    CK(cudaMalloc((void**)&dy, sz));
    CK(cudaMalloc((void**)&dz, sz));
    CK(cudaMalloc((void**)&dvx, sz));
    CK(cudaMalloc((void**)&dvy, sz));
    CK(cudaMalloc((void**)&dvz, sz));
    CK(cudaMalloc((void**)&dm, sz));
    CK(cudaMalloc((void**)&dax, sz));
    CK(cudaMalloc((void**)&day, sz));
    CK(cudaMalloc((void**)&daz, sz));
    CK(cudaMalloc((void**)&dnax, sz));
    CK(cudaMalloc((void**)&dnay, sz));
    CK(cudaMalloc((void**)&dnaz, sz));

    CK(cudaMemcpy(dx, hx, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, hy, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dz, hz, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvx, hvx, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvy, hvy, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvz, hvz, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm, hm, sz, cudaMemcpyHostToDevice));

    int blk = (n + tile_sz - 1) / tile_sz;
    size_t shm = 4 * tile_sz * sizeof(float);

    /* Initial forces */
    forces_kernel<<<blk, tile_sz, shm>>>(dx, dy, dz, dm, dax, day, daz, n, tile_sz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    for (int t = 0; t < TIMESTEPS; t++) {
        /* Step 1: update positions */
        verlet_pos_kernel<<<blk, tile_sz>>>(dx, dy, dz, dvx, dvy, dvz, dax, day, daz, n, DT);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        /* Step 2: new forces */
        forces_kernel<<<blk, tile_sz, shm>>>(dx, dy, dz, dm, dnax, dnay, dnaz, n, tile_sz);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        /* Step 3: update velocities */
        verlet_vel_kernel<<<blk, tile_sz>>>(dvx, dvy, dvz, dax, day, daz, dnax, dnay, dnaz, n, DT);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        /* Swap old/new acc pointers */
        float* tmp;
        tmp = dax;
        dax = dnax;
        dnax = tmp;
        tmp = day;
        day = dnay;
        dnay = tmp;
        tmp = daz;
        daz = dnaz;
        dnaz = tmp;
    }

    CK(cudaMemcpy(hx, dx, sz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy, dy, sz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hz, dz, sz, cudaMemcpyDeviceToHost));

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(dvx);
    cudaFree(dvy);
    cudaFree(dvz);
    cudaFree(dm);
    cudaFree(dax);
    cudaFree(day);
    cudaFree(daz);
    cudaFree(dnax);
    cudaFree(dnay);
    cudaFree(dnaz);
}

static const char* bk = "CUDA GPU (dynamic tiled shared mem, Velocity Verlet)";

#else /* CPU fallback */

static void run_cuda(float* x, float* y, float* z, float* vx, float* vy, float* vz, float* m, int n, int tile_sz) {
    (void)tile_sz;
    float *ax = (float*)calloc(n, 4), *ay = (float*)calloc(n, 4), *az = (float*)calloc(n, 4);
    float *nax = (float*)calloc(n, 4), *nay = (float*)calloc(n, 4), *naz = (float*)calloc(n, 4);
    compute_forces_seq(x, y, z, m, n, ax, ay, az);
    for (int t = 0; t < TIMESTEPS; t++) {
        float dt2 = 0.5f * DT * DT;
        for (int i = 0; i < n; i++) {
            x[i] += vx[i] * DT + ax[i] * dt2;
            y[i] += vy[i] * DT + ay[i] * dt2;
            z[i] += vz[i] * DT + az[i] * dt2;
        }
        compute_forces_seq(x, y, z, m, n, nax, nay, naz);
        for (int i = 0; i < n; i++) {
            vx[i] += 0.5f * (ax[i] + nax[i]) * DT;
            vy[i] += 0.5f * (ay[i] + nay[i]) * DT;
            vz[i] += 0.5f * (az[i] + naz[i]) * DT;
        }
        float* tmp;
        tmp = ax;
        ax = nax;
        nax = tmp;
        tmp = ay;
        ay = nay;
        nay = tmp;
        tmp = az;
        az = naz;
        naz = tmp;
    }
    free(ax);
    free(ay);
    free(az);
    free(nax);
    free(nay);
    free(naz);
}
static const char* bk = "CPU fallback (Velocity Verlet)";
#endif

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : N_PARTICLES;
    int tile_sz = argc > 2 ? atoi(argv[2]) : 256; /* runtime tile size — like OMP_NUM_THREADS */

    float *x = (float*)malloc(n * 4), *y = (float*)malloc(n * 4), *z = (float*)malloc(n * 4);
    float *vx = (float*)malloc(n * 4), *vy = (float*)malloc(n * 4), *vz = (float*)malloc(n * 4), *m = (float*)malloc(n * 4);
    particles_init(x, y, z, vx, vy, vz, m, n, 42);
    printf("CUDA N-Body [%s]: N=%d T=%d tile=%d\n", bk, n, TIMESTEPS, tile_sz);
    double t0 = now_sec();
    run_cuda(x, y, z, vx, vy, vz, m, n, tile_sz);
    double el = now_sec() - t0;
    printf("Time: %.6f s\nTIME: %.6f\nCHECK: %.6f\n", el, el, particles_checksum(x, y, z, n));
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(m);
    return 0;
}
