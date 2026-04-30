#include "nbody.h"
static void step(float*x,float*y,float*z,float*vx,float*vy,float*vz,float*m,int n){
    for(int i=0;i<n;i++){
        float fx=0,fy=0,fz=0;
        for(int j=0;j<n;j++){
            float dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
            float d2=dx*dx+dy*dy+dz*dz+EPS2;
            float inv=1.0f/sqrtf(d2); float f=m[j]*inv*inv*inv;
            fx+=f*dx;fy+=f*dy;fz+=f*dz;
        }
        vx[i]+=DT*fx;vy[i]+=DT*fy;vz[i]+=DT*fz;
    }
    for(int i=0;i<n;i++){x[i]+=DT*vx[i];y[i]+=DT*vy[i];z[i]+=DT*vz[i];}
}
int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):N_PARTICLES;
    float*x=malloc(n*4),*y=malloc(n*4),*z=malloc(n*4);
    float*vx=malloc(n*4),*vy=malloc(n*4),*vz=malloc(n*4),*m=malloc(n*4);
    particles_init(x,y,z,vx,vy,vz,m,n,42);
    printf("Sequential N-Body: N=%d T=%d\n",n,TIMESTEPS);
    double t0=now_sec();
    for(int t=0;t<TIMESTEPS;t++) step(x,y,z,vx,vy,vz,m,n);
    double el=now_sec()-t0;
    printf("Time: %.6f s\nTIME: %.6f\nCHECK: %.6f\n",el,el,particles_checksum(x,y,z,n));
    free(x);free(y);free(z);free(vx);free(vy);free(vz);free(m); return 0;
}
