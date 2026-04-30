#include "nbody.h"
#include <mpi.h>
int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int n=argc>1?atoi(argv[1]):N_PARTICLES;
    int chunk=n/size; n=chunk*size;
    float*x=malloc(n*4),*y=malloc(n*4),*z=malloc(n*4);
    float*vx=malloc(n*4),*vy=malloc(n*4),*vz=malloc(n*4);
    float*m=malloc(n*4),*pos=malloc(n*3*4);
    if(rank==0){particles_init(x,y,z,vx,vy,vz,m,n,42);printf("MPI N-Body: N=%d T=%d ranks=%d\n",n,TIMESTEPS,size);}
    MPI_Bcast(x,n,MPI_FLOAT,0,MPI_COMM_WORLD); MPI_Bcast(y,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(z,n,MPI_FLOAT,0,MPI_COMM_WORLD); MPI_Bcast(vx,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(vy,n,MPI_FLOAT,0,MPI_COMM_WORLD); MPI_Bcast(vz,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(m,n,MPI_FLOAT,0,MPI_COMM_WORLD);
    int lo=rank*chunk,hi=lo+chunk;
    MPI_Barrier(MPI_COMM_WORLD); double t0=MPI_Wtime();
    for(int t=0;t<TIMESTEPS;t++){
        for(int i=lo;i<hi;i++){
            float fx=0,fy=0,fz=0;
            for(int j=0;j<n;j++){
                float dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
                float d2=dx*dx+dy*dy+dz*dz+EPS2;
                float inv=1.0f/sqrtf(d2); float f=m[j]*inv*inv*inv;
                fx+=f*dx;fy+=f*dy;fz+=f*dz;
            }
            vx[i]+=DT*fx;vy[i]+=DT*fy;vz[i]+=DT*fz;
        }
        for(int i=lo;i<hi;i++){x[i]+=DT*vx[i];y[i]+=DT*vy[i];z[i]+=DT*vz[i];}
        for(int i=lo;i<hi;i++){pos[i*3]=x[i];pos[i*3+1]=y[i];pos[i*3+2]=z[i];}
        MPI_Allgather(MPI_IN_PLACE,chunk*3,MPI_FLOAT,pos,chunk*3,MPI_FLOAT,MPI_COMM_WORLD);
        for(int i=0;i<n;i++){x[i]=pos[i*3];y[i]=pos[i*3+1];z[i]=pos[i*3+2];}
    }
    MPI_Barrier(MPI_COMM_WORLD); double el=MPI_Wtime()-t0;
    if(rank==0) printf("Time: %.6f s\nTIME: %.6f\nCHECK: %.6f\n",el,el,particles_checksum(x,y,z,n));
    free(x);free(y);free(z);free(vx);free(vy);free(vz);free(m);free(pos);
    MPI_Finalize(); return 0;
}
