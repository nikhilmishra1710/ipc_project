#include "nbody.h"
void particles_init(float *x,float *y,float *z,float *vx,float *vy,float *vz,float *m,int n,unsigned seed){
    srand(seed);
    for(int i=0;i<n;i++){
        x[i]=((float)rand()/RAND_MAX-0.5f)*200.0f;
        y[i]=((float)rand()/RAND_MAX-0.5f)*200.0f;
        z[i]=((float)rand()/RAND_MAX-0.5f)*200.0f;
        vx[i]=((float)rand()/RAND_MAX-0.5f)*2.0f;
        vy[i]=((float)rand()/RAND_MAX-0.5f)*2.0f;
        vz[i]=((float)rand()/RAND_MAX-0.5f)*2.0f;
        m[i]=1.0f+((float)rand()/RAND_MAX)*9.0f;
    }
}
float particles_checksum(const float *x,const float *y,const float *z,int n){
    double s=0; for(int i=0;i<n;i++) s+=x[i]+y[i]+z[i]; return (float)s;
}
