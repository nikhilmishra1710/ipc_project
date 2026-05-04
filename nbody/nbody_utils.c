#include "nbody.h"

/* -----------------------------------------------------------------------
 * Galaxy disk initialisation (two-pass).
 *
 * Pass 1 — assign positions (uniform-area disk, thin Gaussian z) and masses.
 * Pass 2 — assign Keplerian circular velocities from enclosed-mass model.
 * ----------------------------------------------------------------------- */
void particles_init(float *x,float *y,float *z,
                    float *vx,float *vy,float *vz,
                    float *m,int n,unsigned seed)
{
    srand(seed);
    float total_mass = 0.0f;

    /* Pass 1 — positions & masses */
    for(int i=0;i<n;i++){
        /* Uniform disk area: r = R * sqrt(u) */
        float r     = GALAXY_R * sqrtf((float)rand()/RAND_MAX);
        float theta = 2.0f*3.14159265f*(float)rand()/RAND_MAX;

        /* Thin disk: z ~ Gaussian(0, 0.5) via Box-Muller */
        float u1    = ((float)rand()+1.0f)/(RAND_MAX+2.0f);
        float u2    = (float)rand()/RAND_MAX;
        float zg    = 0.5f*sqrtf(-2.0f*logf(u1))*cosf(2.0f*3.14159265f*u2);

        x[i] = r*cosf(theta);
        y[i] = r*sinf(theta);
        z[i] = zg;
        m[i] = 1e10f + ((float)rand()/RAND_MAX)*9e10f;
        total_mass += m[i];
    }

    /* Pass 2 — Keplerian circular velocities */
    for(int i=0;i<n;i++){
        float r     = sqrtf(x[i]*x[i]+y[i]*y[i]);
        float theta = atan2f(y[i],x[i]);
        /* Enclosed mass for uniform disk: M_enc = M_total*(r/R)^2 */
        float frac  = r/GALAXY_R;
        float M_enc = total_mass*frac*frac;
        float v_c   = (r>0.1f) ? sqrtf(G*M_enc/r) : 0.0f;
        vx[i] = -v_c*sinf(theta);   /* tangential — perpendicular to r */
        vy[i] =  v_c*cosf(theta);
        vz[i] =  0.0f;
    }
}

/* Checksum — sum of all coordinate values (reproducibility check) */
float particles_checksum(const float *x,const float *y,const float *z,int n){
    double s=0;
    for(int i=0;i<n;i++) s+=x[i]+y[i]+z[i];
    return (float)s;
}

/* -----------------------------------------------------------------------
 * Shared force-computation kernel (sequential / used by OMP via pragma).
 * Writes per-particle acceleration into ax/ay/az.
 * ----------------------------------------------------------------------- */
void compute_forces_seq(const float * __restrict__ x,
                        const float * __restrict__ y,
                        const float * __restrict__ z,
                        const float * __restrict__ m,
                        int n,
                        float * __restrict__ ax,
                        float * __restrict__ ay,
                        float * __restrict__ az)
{
    for(int i=0;i<n;i++){
        float fx=0,fy=0,fz=0;
        float xi=x[i],yi=y[i],zi=z[i];
        for(int j=0;j<n;j++){
            float dx=x[j]-xi, dy=y[j]-yi, dz=z[j]-zi;
            float d2=dx*dx+dy*dy+dz*dz+EPS2;
            float inv=1.0f/sqrtf(d2);
            float f=G*m[j]*inv*inv*inv;
            fx+=f*dx; fy+=f*dy; fz+=f*dz;
        }
        ax[i]=fx; ay[i]=fy; az[i]=fz;
    }
}
