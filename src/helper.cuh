#pragma once
#include "class.cuh"

#define _USE_MATH_DEFINES
#include <math.h>

void gpu_info(std::shared_ptr<cudaDeviceProp>);

__device__ __host__ 
unsigned _rel_count (const unsigned);

// __device__ __host__ 
// void _rec_idx (unsigned, unsigned&, unsigned& b, const unsigned = 0);

__device__ __host__ 
void _quad_idx (unsigned, unsigned&, unsigned&);

__device__ __host__ 
unsigned _rel_pos (const unsigned, const unsigned);

__global__
void cal_rel_coords (
    Rel_Force_Vector<float>  * const __restrict__ ,
    const Tensors<float>     * const __restrict__ ,
    const float , const float , 
    const float , const float ,
    const unsigned 
);

/*
*   Smoothing Kernels
*/

#define C_POLY6         315 * M_1_PI / 64
#define C_LAP_POLY6     C_POLY6 * 6
#define C_GRAD_SPIKY     45 * M_1_PI 
#define C_LAP_VISC      C_GRAD_SPIKY

// W_poly6 for density 
template <class Typ>
__device__ __host__
Typ W_poly6 (
    const Typ& ,
    const Typ& ,
    const Typ& ,
    const Typ& 
);

// lap_W_poly6 surface tension 
template <class Typ>
__device__ __host__
Typ lap_W_poly6 (
    const Typ& ,
    const Typ& ,
    const Typ& ,
    const Typ& 
);

// grad_W_spiky for pressure
template <class Typ>
__device__ __host__
Typ grad_W_spiky (
    const Typ& ,
    const Typ& ,
    const Typ& 
);

// lap_W_vis for Viscosity
template <class Typ>
__device__ __host__
Typ lap_W_viscosity (
    const Typ& ,
    const Typ& ,
    const Typ& 
);
