#include "bloom.h"
#include <mpi.h>
int main(int argc,char**argv){
    MPI_Init(&argc,&argv); int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int n=argc>1?atoi(argv[1]):5000000; int chunk=n/size; n=chunk*size;
    char(*urls)[URL_LEN]=malloc((size_t)n*sizeof(*urls));
    if(rank==0){generate_urls(urls,n,42);printf("MPI Bloom: n=%d ranks=%d\n",n,size);}
    char(*local)[URL_LEN]=malloc((size_t)chunk*sizeof(*local));
    MPI_Scatter(urls,chunk*URL_LEN,MPI_BYTE,local,chunk*URL_LEN,MPI_BYTE,0,MPI_COMM_WORLD);
    BloomFilter bf; bloom_init(&bf,BLOOM_M,BLOOM_K);
    MPI_Barrier(MPI_COMM_WORLD); double t0=MPI_Wtime();
    for(int i=0;i<chunk;i++) bloom_insert(&bf,local[i]);
    uint8_t *merged=(uint8_t*)calloc(BLOOM_BYTES,1);
    MPI_Allreduce(bf.bits,merged,BLOOM_BYTES,MPI_BYTE,MPI_BOR,MPI_COMM_WORLD);
    memcpy(bf.bits,merged,BLOOM_BYTES); free(merged);
    MPI_Barrier(MPI_COMM_WORLD); double el=MPI_Wtime()-t0;
    if(rank==0){
        long bs=count_set_bits(&bf);
        int fn=0; for(int i=0;i<N_QUERY&&i<n;i++) if(!bloom_query(&bf,urls[i])) fn++;
        char(*q)[URL_LEN]=malloc((size_t)N_QUERY*sizeof(*q)); generate_urls(q,N_QUERY,9999);
        int fp=0; for(int i=0;i<N_QUERY;i++) if(bloom_query(&bf,q[i])) fp++;
        double fpr=(double)fp/N_QUERY;
        printf("Time: %.6f s\nFN: %d\nFPR: %.6f\n",el,fn,fpr);
        printf("TIME: %.6f\nFPR_V: %.6f\nFN_V: %d\nBITS_V: %ld\n",el,fpr,fn,bs);
        free(q);
    }
    bloom_destroy(&bf); free(urls); free(local); MPI_Finalize(); return 0;
}
