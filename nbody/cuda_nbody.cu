#include "nbody.h"
#ifndef NO_CUDA
#include <cuda_runtime.h>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__global__ void forces_kernel(float *x,float *y,float *z,float *vx,float *vy,float *vz,float *mass,int n,float dt){
    __shared__ float sx[TILE_SIZE],sy[TILE_SIZE],sz[TILE_SIZE],sm[TILE_SIZE];
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    float px=0,py=0,pz=0,fx=0,fy=0,fz=0;
    if(i<n){px=x[i];py=y[i];pz=z[i];}
    for(int tile=0;tile<n;tile+=TILE_SIZE){
        int j=tile+threadIdx.x;
        sx[threadIdx.x]=j<n?x[j]:0; sy[threadIdx.x]=j<n?y[j]:0;
        sz[threadIdx.x]=j<n?z[j]:0; sm[threadIdx.x]=j<n?mass[j]:0;
        __syncthreads();
        if(i<n){
            for(int k=0;k<TILE_SIZE;k++){
                float dx=sx[k]-px,dy=sy[k]-py,dz=sz[k]-pz;
                float d2=dx*dx+dy*dy+dz*dz+EPS2;
                float inv=rsqrtf(d2);
                float f=sm[k]*inv*inv*inv;
                fx+=f*dx; fy+=f*dy; fz+=f*dz;
            }
        }
        __syncthreads();
    }
    if(i<n){vx[i]+=dt*fx; vy[i]+=dt*fy; vz[i]+=dt*fz;}
}

__global__ void update_kernel(float *x,float *y,float *z,float *vx,float *vy,float *vz,int n,float dt){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){x[i]+=dt*vx[i]; y[i]+=dt*vy[i]; z[i]+=dt*vz[i];}
}

static void run_cuda(float *hx,float *hy,float *hz,float *hvx,float *hvy,float *hvz,float *hm,int n){
    size_t sz=n*sizeof(float);
    float *dx,*dy,*dz,*dvx,*dvy,*dvz,*dm;
    CK(cudaMalloc((void**)&dx,sz)); CK(cudaMalloc((void**)&dy,sz)); CK(cudaMalloc((void**)&dz,sz));
    CK(cudaMalloc((void**)&dvx,sz)); CK(cudaMalloc((void**)&dvy,sz)); CK(cudaMalloc((void**)&dvz,sz));
    CK(cudaMalloc((void**)&dm,sz));
    CK(cudaMemcpy(dx,hx,sz,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dy,hy,sz,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dz,hz,sz,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dvx,hvx,sz,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvy,hvy,sz,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dvz,hvz,sz,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm,hm,sz,cudaMemcpyHostToDevice));
    int thr=TILE_SIZE,blk=(n+thr-1)/thr;
    for(int t=0;t<TIMESTEPS;t++){
        forces_kernel<<<blk,thr>>>(dx,dy,dz,dvx,dvy,dvz,dm,n,DT);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        update_kernel<<<blk,thr>>>(dx,dy,dz,dvx,dvy,dvz,n,DT);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpy(hx,dx,sz,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hy,dy,sz,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hz,dz,sz,cudaMemcpyDeviceToHost));
    cudaFree(dx);cudaFree(dy);cudaFree(dz);cudaFree(dvx);cudaFree(dvy);cudaFree(dvz);cudaFree(dm);
}
static const char *bk="CUDA GPU (tiled shared mem)";
#else
static void run_cuda(float *x,float *y,float *z,float *vx,float *vy,float *vz,float *m,int n){
    for(int t=0;t<TIMESTEPS;t++){
        for(int i=0;i<n;i++){
            float fx=0,fy=0,fz=0;
            for(int j=0;j<n;j++){
                float dx2=x[j]-x[i],dy2=y[j]-y[i],dz2=z[j]-z[i];
                float d2=dx2*dx2+dy2*dy2+dz2*dz2+EPS2;
                float inv=1.0f/sqrtf(d2); float f=m[j]*inv*inv*inv;
                fx+=f*dx2;fy+=f*dy2;fz+=f*dz2;
            }
            vx[i]+=DT*fx;vy[i]+=DT*fy;vz[i]+=DT*fz;
        }
        for(int i=0;i<n;i++){x[i]+=DT*vx[i];y[i]+=DT*vy[i];z[i]+=DT*vz[i];}
    }
}
static const char *bk="CPU fallback";
#endif

int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):N_PARTICLES;
    float *x=(float*)malloc(n*4),*y=(float*)malloc(n*4),*z=(float*)malloc(n*4);
    float *vx=(float*)malloc(n*4),*vy=(float*)malloc(n*4),*vz=(float*)malloc(n*4);
    float *m=(float*)malloc(n*4);
    particles_init(x,y,z,vx,vy,vz,m,n,42);
    printf("CUDA N-Body [%s]: N=%d T=%d\n",bk,n,TIMESTEPS);
    double t0=now_sec();
    run_cuda(x,y,z,vx,vy,vz,m,n);
    double elapsed=now_sec()-t0;
    printf("Time: %.6f s\nChecksum: %.6f\n",elapsed,particles_checksum(x,y,z,n));
    printf("TIME: %.6f\nCHECK: %.6f\n",elapsed,particles_checksum(x,y,z,n));
    free(x);free(y);free(z);free(vx);free(vy);free(vz);free(m);
    return 0;
}
