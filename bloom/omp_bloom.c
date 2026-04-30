#include "bloom.h"
#include <omp.h>
int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):5000000, nt;
    #pragma omp parallel
    #pragma omp single
    nt=omp_get_num_threads();
    printf("OpenMP Bloom: n=%d threads=%d\n",n,nt);
    char(*urls)[URL_LEN]=malloc((size_t)n*sizeof(*urls));
    generate_urls(urls,n,42);
    BloomFilter bf; bloom_init(&bf,BLOOM_M,BLOOM_K);
    uint8_t *bits=bf.bits; int k=bf.k;
    double t0=omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++){
        uint32_t h1=murmur3_32(urls[i],0),h2=fnv1a_32(urls[i],1);
        for(int j=0;j<k;j++){size_t bit=((uint64_t)h1+(uint64_t)j*h2)%BLOOM_M;__atomic_or_fetch(&bits[bit>>3],(uint8_t)(1u<<(bit&7)),__ATOMIC_RELAXED);}
    }
    double el=omp_get_wtime()-t0;
    long bs=count_set_bits(&bf);
    int fn=0; for(int i=0;i<N_QUERY&&i<n;i++) if(!bloom_query(&bf,urls[i])) fn++;
    char(*q)[URL_LEN]=malloc((size_t)N_QUERY*sizeof(*q)); generate_urls(q,N_QUERY,9999);
    int fp=0; for(int i=0;i<N_QUERY;i++) if(bloom_query(&bf,q[i])) fp++;
    double fpr=(double)fp/N_QUERY;
    printf("Time: %.6f s\nFN: %d\nFPR: %.6f\n",el,fn,fpr);
    printf("TIME: %.6f\nFPR_V: %.6f\nFN_V: %d\nBITS_V: %ld\n",el,fpr,fn,bs);
    bloom_destroy(&bf); free(urls); free(q); return 0;
}
