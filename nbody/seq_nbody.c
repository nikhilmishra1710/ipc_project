#include "nbody.h"

int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):N_PARTICLES;

    float *x =malloc(n*sizeof(float)), *y =malloc(n*sizeof(float)), *z =malloc(n*sizeof(float));
    float *vx=malloc(n*sizeof(float)), *vy=malloc(n*sizeof(float)), *vz=malloc(n*sizeof(float));
    float *m =malloc(n*sizeof(float));
    /* Velocity-Verlet needs current and new accelerations */
    float *ax =malloc(n*sizeof(float)), *ay =malloc(n*sizeof(float)), *az =malloc(n*sizeof(float));
    float *nax=malloc(n*sizeof(float)), *nay=malloc(n*sizeof(float)), *naz=malloc(n*sizeof(float));

    particles_init(x,y,z,vx,vy,vz,m,n,42);
    printf("Sequential N-Body (Galaxy, Velocity Verlet): N=%d T=%d G=%.3e\n",
           n,TIMESTEPS,(double)G);

    /* Compute initial accelerations */
    compute_forces_seq(x,y,z,m,n,ax,ay,az);

    double t0=now_sec();
    for(int t=0;t<TIMESTEPS;t++){
        /* --- Verlet position update --- */
        for(int i=0;i<n;i++){
            float dt2=0.5f*DT*DT;
            x[i]+=vx[i]*DT+ax[i]*dt2;
            y[i]+=vy[i]*DT+ay[i]*dt2;
            z[i]+=vz[i]*DT+az[i]*dt2;
        }
        /* --- New forces at updated positions --- */
        compute_forces_seq(x,y,z,m,n,nax,nay,naz);
        /* --- Verlet velocity update --- */
        for(int i=0;i<n;i++){
            vx[i]+=0.5f*(ax[i]+nax[i])*DT;
            vy[i]+=0.5f*(ay[i]+nay[i])*DT;
            vz[i]+=0.5f*(az[i]+naz[i])*DT;
        }
        /* Swap acc pointers */
        float *tmp;
        tmp=ax; ax=nax; nax=tmp;
        tmp=ay; ay=nay; nay=tmp;
        tmp=az; az=naz; naz=tmp;
    }
    double el=now_sec()-t0;
    printf("Time: %.6f s\nTIME: %.6f\nCHECK: %.6f\n",el,el,
           particles_checksum(x,y,z,n));

    free(x);free(y);free(z);free(vx);free(vy);free(vz);free(m);
    free(ax);free(ay);free(az);free(nax);free(nay);free(naz);
    return 0;
}
