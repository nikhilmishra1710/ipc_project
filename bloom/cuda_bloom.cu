#include "bloom.h"
#ifndef NO_CUDA
#include <cuda_runtime.h>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__device__ static uint32_t d_murmur3(const char*k,uint32_t s){
    const uint8_t*d=(const uint8_t*)k;int len=0;while(d[len])len++;int nb=len/4;uint32_t h=s,C1=0xcc9e2d51u,C2=0x1b873593u;
    for(int i=0;i<nb;i++){uint32_t k2=((uint32_t)d[i*4])|((uint32_t)d[i*4+1]<<8)|((uint32_t)d[i*4+2]<<16)|((uint32_t)d[i*4+3]<<24);k2*=C1;k2=(k2<<15)|(k2>>17);k2*=C2;h^=k2;h=(h<<13)|(h>>19);h=h*5u+0xe6546b64u;}
    const uint8_t*t=d+nb*4;uint32_t k1=0;switch(len&3){case 3:k1^=(uint32_t)t[2]<<16;case 2:k1^=(uint32_t)t[1]<<8;case 1:k1^=(uint32_t)t[0];k1*=C1;k1=(k1<<15)|(k1>>17);k1*=C2;h^=k1;}
    h^=(uint32_t)len;h^=h>>16;h*=0x85ebca6bu;h^=h>>13;h*=0xc2b2ae35u;h^=h>>16;return h;}
__device__ static uint32_t d_fnv1a(const char*k,uint32_t s){uint32_t h=2166136261u^s;for(const char*p=k;*p;p++){h^=(uint8_t)*p;h*=16777619u;}return h;}
__global__ void bk_insert(const char*urls,int n,unsigned int*words,unsigned long long m,int k){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const char*u=urls+(unsigned long long)i*URL_LEN;
    uint32_t h1=d_murmur3(u,0),h2=d_fnv1a(u,1);
    for(int j=0;j<k;j++){unsigned long long bit=((unsigned long long)h1+(unsigned long long)j*h2)%m;atomicOr(&words[bit/32],1u<<(unsigned)(bit%32));}
}
static void cuda_build(char(*urls)[URL_LEN],BloomFilter*bf,int n){
    unsigned int*dw;CK(cudaMalloc((void**)&dw,BLOOM_BYTES));CK(cudaMemset(dw,0,BLOOM_BYTES));
    char*du;CK(cudaMalloc((void**)&du,(size_t)CUDA_BATCH*URL_LEN));
    int done=0;while(done<n){int b=n-done<CUDA_BATCH?n-done:CUDA_BATCH;
        CK(cudaMemcpy(du,urls[done],(size_t)b*URL_LEN,cudaMemcpyHostToDevice));
        bk_insert<<<(b+255)/256,256>>>(du,b,dw,(unsigned long long)bf->m,bf->k);
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());done+=b;}
    CK(cudaMemcpy(bf->bits,dw,BLOOM_BYTES,cudaMemcpyDeviceToHost));cudaFree(dw);cudaFree(du);
}
static const char*bk_name="CUDA GPU";
#else
static void cuda_build(char(*urls)[URL_LEN],BloomFilter*bf,int n){for(int i=0;i<n;i++)bloom_insert(bf,urls[i]);}
static const char*bk_name="CPU fallback";
#endif
int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):5000000;
    printf("CUDA Bloom [%s]: n=%d\n",bk_name,n);
    char(*urls)[URL_LEN]=(char(*)[URL_LEN])malloc((size_t)n*sizeof(*urls));
    generate_urls(urls,n,42);
    BloomFilter bf; bloom_init(&bf,BLOOM_M,BLOOM_K);
    double t0=now_sec(); cuda_build(urls,&bf,n); double el=now_sec()-t0;
    long bs=count_set_bits(&bf);
    int fn=0; for(int i=0;i<N_QUERY&&i<n;i++) if(!bloom_query(&bf,urls[i])) fn++;
    char(*q)[URL_LEN]=(char(*)[URL_LEN])malloc((size_t)N_QUERY*sizeof(*q)); generate_urls(q,N_QUERY,9999);
    int fp=0; for(int i=0;i<N_QUERY;i++) if(bloom_query(&bf,q[i])) fp++;
    double fpr=(double)fp/N_QUERY;
    printf("TIME: %.6f\nFPR_V: %.6f\nFN_V: %d\nBITS_V: %ld\n",el,fpr,fn,bs);
    bloom_destroy(&bf); free(urls); free(q); return 0;
}
