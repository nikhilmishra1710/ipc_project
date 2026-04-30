#include "bloom.h"
int bloom_init(BloomFilter *bf,size_t m,int k){bf->bits=(uint8_t*)calloc(m/8,1);if(!bf->bits)return -1;bf->m=m;bf->k=k;return 0;}
void bloom_destroy(BloomFilter *bf){free(bf->bits);bf->bits=NULL;}
uint32_t murmur3_32(const char *key,uint32_t seed){
    const uint8_t *d=(const uint8_t*)key;size_t len=strlen(key),nb=len/4;
    uint32_t h=seed,C1=0xcc9e2d51u,C2=0x1b873593u;
    for(size_t i=0;i<nb;i++){uint32_t k;memcpy(&k,d+i*4,4);k*=C1;k=(k<<15)|(k>>17);k*=C2;h^=k;h=(h<<13)|(h>>19);h=h*5u+0xe6546b64u;}
    const uint8_t *t=d+nb*4;uint32_t k1=0;
    switch(len&3){case 3:k1^=(uint32_t)t[2]<<16;case 2:k1^=(uint32_t)t[1]<<8;case 1:k1^=(uint32_t)t[0];k1*=C1;k1=(k1<<15)|(k1>>17);k1*=C2;h^=k1;}
    h^=(uint32_t)len;h^=h>>16;h*=0x85ebca6bu;h^=h>>13;h*=0xc2b2ae35u;h^=h>>16;return h;
}
uint32_t fnv1a_32(const char *key,uint32_t seed){uint32_t h=2166136261u^seed;for(const char *p=key;*p;p++){h^=(uint8_t)*p;h*=16777619u;}return h;}
void bloom_insert(BloomFilter *bf,const char *item){
    uint32_t h1=murmur3_32(item,0),h2=fnv1a_32(item,1);
    for(int j=0;j<bf->k;j++){size_t bit=((uint64_t)h1+(uint64_t)j*h2)%bf->m;bf->bits[bit>>3]|=(uint8_t)(1u<<(bit&7));}
}
int bloom_query(const BloomFilter *bf,const char *item){
    uint32_t h1=murmur3_32(item,0),h2=fnv1a_32(item,1);
    for(int j=0;j<bf->k;j++){size_t bit=((uint64_t)h1+(uint64_t)j*h2)%bf->m;if(!(bf->bits[bit>>3]&(uint8_t)(1u<<(bit&7))))return 0;}return 1;
}
void generate_urls(char(*u)[URL_LEN],int n,unsigned s){srand(s);for(int i=0;i<n;i++)snprintf(u[i],URL_LEN,"http://url-%010d-%06x.io",i,(unsigned)(rand()&0xFFFFFF));}
double calc_fp_rate(size_t n,size_t m,int k){return pow(1.0-exp(-(double)k*(double)n/(double)m),(double)k);}
long count_set_bits(const BloomFilter *bf){long c=0;for(size_t i=0;i<bf->m/8;i++)c+=__builtin_popcount(bf->bits[i]);return c;}
