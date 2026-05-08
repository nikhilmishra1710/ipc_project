#include "nbody.h"
#include <omp.h>

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : N_PARTICLES, nt;

    float *x = malloc(n * sizeof(float)), *y = malloc(n * sizeof(float)), *z = malloc(n * sizeof(float));
    float *vx = malloc(n * sizeof(float)), *vy = malloc(n * sizeof(float)), *vz = malloc(n * sizeof(float));
    float* m = malloc(n * sizeof(float));
    float *ax = malloc(n * sizeof(float)), *ay = malloc(n * sizeof(float)), *az = malloc(n * sizeof(float));
    float *nax = malloc(n * sizeof(float)), *nay = malloc(n * sizeof(float)), *naz = malloc(n * sizeof(float));

    particles_init(x, y, z, vx, vy, vz, m, n, 42);

#pragma omp parallel
#pragma omp single
    nt = omp_get_num_threads();
    printf("OpenMP N-Body (Galaxy, Velocity Verlet): N=%d T=%d threads=%d G=%.3e\n", n, TIMESTEPS, nt, (double)G);

/* Initial accelerations — parallelised */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float fx = 0, fy = 0, fz = 0, xi = x[i], yi = y[i], zi = z[i];
        for (int j = 0; j < n; j++) {
            float dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            float d2 = dx * dx + dy * dy + dz * dz + EPS2;
            float inv = 1.0f / sqrtf(d2);
            float f = G * m[j] * inv * inv * inv;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        }
        ax[i] = fx;
        ay[i] = fy;
        az[i] = fz;
    }

    double t0 = omp_get_wtime();
#pragma omp parallel
    {
        for (int t = 0; t < TIMESTEPS; t++) {

/* --- Verlet position update --- */
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                float dt2 = 0.5f * DT * DT;
                x[i] += vx[i] * DT + ax[i] * dt2;
                y[i] += vy[i] * DT + ay[i] * dt2;
                z[i] += vz[i] * DT + az[i] * dt2;
            }
/* implicit barrier — positions consistent before force pass */

/* --- New forces at updated positions --- */
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                float fx = 0, fy = 0, fz = 0, xi = x[i], yi = y[i], zi = z[i];
                for (int j = 0; j < n; j++) {
                    float dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
                    float d2 = dx * dx + dy * dy + dz * dz + EPS2;
                    float inv = 1.0f / sqrtf(d2);
                    float f = G * m[j] * inv * inv * inv;
                    fx += f * dx;
                    fy += f * dy;
                    fz += f * dz;
                }
                nax[i] = fx;
                nay[i] = fy;
                naz[i] = fz;
            }
/* implicit barrier */

/* --- Verlet velocity update --- */
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                vx[i] += 0.5f * (ax[i] + nax[i]) * DT;
                vy[i] += 0.5f * (ay[i] + nay[i]) * DT;
                vz[i] += 0.5f * (az[i] + naz[i]) * DT;
            }
/* implicit barrier */

/* Swap acc arrays (single thread, others wait at next omp for) */
#pragma omp single
            {
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
            /* implicit barrier after single */
        }
    }
    double el = omp_get_wtime() - t0;
    printf("Time: %.6f s\nTIME: %.6f\nCHECK: %.6f\n", el, el, particles_checksum(x, y, z, n));

    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(m);
    free(ax);
    free(ay);
    free(az);
    free(nax);
    free(nay);
    free(naz);
    return 0;
}
