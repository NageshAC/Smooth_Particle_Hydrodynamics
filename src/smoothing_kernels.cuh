#pragma once
#include "cuda_runtime.h"

#define _USE_MATH_DEFINES
#include <math.h>

#define C_POLY6         315 * M_1_PI / 64
#define C_GRAD_POLY6    945 * M_1_PI / 32
#define C_SPIKY          15 * M_1_PI 
#define C_GRAD_SPIKY     45 * M_1_PI 
#define C_LAP_VISC      C_GRAD_SPIKY

__device__ __host__
inline double W_poly6 (
    const double& , 
    const double& , 
    const double&
);

__device__ __host__
inline double grad_W_poly6 (
    const double& , 
    const double& , 
    const double& , 
    const double& 
);

__device__ __host__
inline double W_spiky (
    const double& ,
    const double& ,
    const double& 
);

__device__ __host__
inline double grad_W_spiky (
    const double& ,
    const double& ,
    const double& 
);

__device__ __host__
inline double lap_W_viscosity (
    const double& ,
    const double& ,
    const double& 
);
