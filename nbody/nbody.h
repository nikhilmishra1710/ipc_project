#ifndef NBODY_H
#define NBODY_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_PARTICLES 30000
#define TIMESTEPS   5
#define DT          0.01f
#define GALAXY_R    100.0f    /* outer radius of galaxy disk (m) */
#define EPS2        0.01f
#define TILE_SIZE   256

/* Gravitational constant — SI value (N·m²/kg²).
   Particle masses are in kg, positions in metres.
   Scale masses/positions in particles_init if you want different units. */
#define G           6.674e-11f

/* Barnes-Hut opening-angle criterion */
#define BH_THETA    0.5f

#ifdef __cplusplus
extern "C" {
#endif
/* Initialise particles as a rotating galaxy disk */
void   particles_init(float *x,float *y,float *z,
                      float *vx,float *vy,float *vz,
                      float *m,int n,unsigned seed);
float  particles_checksum(const float *x,const float *y,const float *z,int n);
/* Compute net force (acceleration) on every particle into ax/ay/az arrays */
void   compute_forces_seq(const float *x,const float *y,const float *z,
                          const float *m,int n,
                          float *ax,float *ay,float *az);
#ifdef __cplusplus
}
#endif

static inline double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec+ts.tv_nsec*1e-9;
}
#endif
