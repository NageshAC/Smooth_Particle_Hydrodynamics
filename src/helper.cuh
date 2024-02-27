#pragma once
#include <cuda_runtime.h>
#include "thrust/device_vector.h"

// i, j
#define depth_rel_idx 2

void gpu_info(std::shared_ptr<cudaDeviceProp>);

__device__ __host__
unsigned _rel_count (const unsigned &);

__host__
dim3 _get_block_size (
    const unsigned &, 
    std::shared_ptr<cudaDeviceProp>
);

__host__
unsigned _get_thread_size (
    const unsigned &, 
    std::shared_ptr<cudaDeviceProp>
);

__global__
void cal_rel_idx (
    unsigned * const,
    const unsigned & 
);


// unsigned _idx3D (
//     const unsigned , /     const unsigned & , 
//     const unsigned , 
//     const unsigned & , 
//     const unsigned 
// );

// unsigned _idx2D (
//     const unsigned , 
//     const unsigned & , 
//     const unsigned 
// );