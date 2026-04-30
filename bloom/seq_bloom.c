#include "bloom.h"
int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):5000000;
    printf("Sequential Bloom: n=%d m=%llu k=%d\n",n,(unsigned long long)BLOOM_M,BLOOM_K);
    char(*urls)[URL_LEN]=malloc((size_t)n*sizeof(*urls));
    generate_urls(urls,n,42);
    BloomFilter bf; bloom_init(&bf,BLOOM_M,BLOOM_K);
    double t0=now_sec();
    for(int i=0;i<n;i++) bloom_insert(&bf,urls[i]);
    double el=now_sec()-t0;
    long bits=count_set_bits(&bf);
    int fn=0; for(int i=0;i<N_QUERY&&i<n;i++) if(!bloom_query(&bf,urls[i])) fn++;
    char(*q)[URL_LEN]=malloc((size_t)N_QUERY*sizeof(*q)); generate_urls(q,N_QUERY,9999);
    int fp=0; for(int i=0;i<N_QUERY;i++) if(bloom_query(&bf,q[i])) fp++;
    double fpr=(double)fp/N_QUERY;
    printf("Time: %.6f s\nBits: %ld/%llu\nFN: %d\nFPR: %.6f\n",el,bits,(unsigned long long)BLOOM_M,fn,fpr);
    printf("TIME: %.6f\nFPR_V: %.6f\nFN_V: %d\nBITS_V: %ld\n",el,fpr,fn,bits);
    bloom_destroy(&bf); free(urls); free(q); return 0;
}
