#include "nbody.h"
#include <mpi.h>

/* Inline force accumulation for one particle i against all n */
static void force_on(int i,const float *x,const float *y,const float *z,
                     const float *m,int n,float *fx,float *fy,float *fz){
    float lx=0,ly=0,lz=0,xi=x[i],yi=y[i],zi=z[i];
    for(int j=0;j<n;j++){
        float dx=x[j]-xi,dy=y[j]-yi,dz=z[j]-zi;
        float d2=dx*dx+dy*dy+dz*dz+EPS2;
        float inv=1.0f/sqrtf(d2);
        float f=G*m[j]*inv*inv*inv;
        lx+=f*dx; ly+=f*dy; lz+=f*dz;
    }
    *fx=lx; *fy=ly; *fz=lz;
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int n=argc>1?atoi(argv[1]):N_PARTICLES;
    int chunk=n/size; n=chunk*size;
    int lo=rank*chunk, hi=lo+chunk;

    float *x =malloc(n*sizeof(float)), *y =malloc(n*sizeof(float)), *z =malloc(n*sizeof(float));
    float *vx=malloc(n*sizeof(float)), *vy=malloc(n*sizeof(float)), *vz=malloc(n*sizeof(float));
    float *m =malloc(n*sizeof(float));
    /* Local slices for current and new accelerations */
    float *ax =calloc(chunk,sizeof(float)), *ay =calloc(chunk,sizeof(float)), *az =calloc(chunk,sizeof(float));
    float *nax=calloc(chunk,sizeof(float)), *nay=calloc(chunk,sizeof(float)), *naz=calloc(chunk,sizeof(float));
    /* Position sync buffer (3 floats per particle) */
    float *pbuf=malloc(n*3*sizeof(float));

    if(rank==0){
        particles_init(x,y,z,vx,vy,vz,m,n,42);
        printf("MPI N-Body (Galaxy, Velocity Verlet): N=%d T=%d ranks=%d G=%.3e\n",
               n,TIMESTEPS,size,(double)G);
    }
    MPI_Bcast(x,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(y,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(z,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(vx,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(vy,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(vz,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(m,n,MPI_FLOAT,0,MPI_COMM_WORLD);

    /* Initial accelerations for this rank's slice */
    for(int i=lo;i<hi;i++){
        int li=i-lo;
        force_on(i,x,y,z,m,n,&ax[li],&ay[li],&az[li]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0=MPI_Wtime();

    for(int t=0;t<TIMESTEPS;t++){
        /* --- Verlet position update for slice --- */
        float dt2=0.5f*DT*DT;
        for(int i=lo;i<hi;i++){
            int li=i-lo;
            x[i]+=vx[i]*DT+ax[li]*dt2;
            y[i]+=vy[i]*DT+ay[li]*dt2;
            z[i]+=vz[i]*DT+az[li]*dt2;
        }

        /* --- Sync positions only (Verlet forces depend only on positions) --- */
        for(int i=lo;i<hi;i++){
            int b=(i-lo)*3;
            pbuf[lo*3+b+0]=x[i]; pbuf[lo*3+b+1]=y[i]; pbuf[lo*3+b+2]=z[i];
        }
        MPI_Allgather(MPI_IN_PLACE,chunk*3,MPI_FLOAT,pbuf,chunk*3,MPI_FLOAT,MPI_COMM_WORLD);
        for(int i=0;i<n;i++){
            x[i]=pbuf[i*3+0]; y[i]=pbuf[i*3+1]; z[i]=pbuf[i*3+2];
        }

        /* --- New forces at updated positions --- */
        for(int i=lo;i<hi;i++){
            int li=i-lo;
            force_on(i,x,y,z,m,n,&nax[li],&nay[li],&naz[li]);
        }

        /* --- Verlet velocity update --- */
        for(int i=lo;i<hi;i++){
            int li=i-lo;
            vx[i]+=0.5f*(ax[li]+nax[li])*DT;
            vy[i]+=0.5f*(ay[li]+nay[li])*DT;
            vz[i]+=0.5f*(az[li]+naz[li])*DT;
        }

        /* Swap acc arrays */
        float *tmp;
        tmp=ax; ax=nax; nax=tmp;
        tmp=ay; ay=nay; nay=tmp;
        tmp=az; az=naz; naz=tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double el=MPI_Wtime()-t0;

    if(rank==0)
        printf("Time: %.6f s\nTIME: %.6f\nCHECK: %.6f\n",
               el,el,particles_checksum(x,y,z,n));

    free(x);free(y);free(z);free(vx);free(vy);free(vz);free(m);
    free(ax);free(ay);free(az);free(nax);free(nay);free(naz);free(pbuf);
    MPI_Finalize();
    return 0;
}
