#ifndef NBODY_H
#define NBODY_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_PARTICLES 30000
#define TIMESTEPS   10
#define DT          0.01f
#define EPS2        0.01f
#define TILE_SIZE   256

#ifdef __cplusplus
extern "C" {
#endif
void   particles_init(float *x,float *y,float *z,float *vx,float *vy,float *vz,float *m,int n,unsigned seed);
float  particles_checksum(const float *x,const float *y,const float *z,int n);
#ifdef __cplusplus
}
#endif

static inline double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec+ts.tv_nsec*1e-9;
}
#endif
