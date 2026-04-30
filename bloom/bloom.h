#ifndef BLOOM_H
#define BLOOM_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define BLOOM_M     67108864ULL
#define BLOOM_BYTES (BLOOM_M/8)
#define BLOOM_K     7
#define URL_LEN     48
#define N_QUERY     500000
#define CUDA_BATCH  500000
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint8_t *bits; size_t m; int k; } BloomFilter;
int      bloom_init(BloomFilter*,size_t,int);
void     bloom_destroy(BloomFilter*);
uint32_t murmur3_32(const char*,uint32_t);
uint32_t fnv1a_32(const char*,uint32_t);
void     bloom_insert(BloomFilter*,const char*);
int      bloom_query(const BloomFilter*,const char*);
void     generate_urls(char(*)[URL_LEN],int,unsigned);
double   calc_fp_rate(size_t,size_t,int);
long     count_set_bits(const BloomFilter*);
#ifdef __cplusplus
}
#endif
static inline double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#endif
