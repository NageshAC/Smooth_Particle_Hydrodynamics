#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "smoothing_kernels.cuh"

// rel_x, rel_y, rel_z, distance
#define depth_rel_pos 4

// __global__
// void cal_rel_pos (
//     thrust::device_vector<double>& rel_pos,
//     const thrust::device_vector<double>& pos
// );

// __global__
// void cal_density (
//     thrust::device_vector<double>& density,
//     const double& mass,
//     const thrust::device_vector<thrust::device_vector<thrust::device_vector<double>>>& rel_pos,
//     const double& C_h2, const double& C_h9
// );

// __global__
// void cal_pressure (
//     thrust::device_vector<thrust::device_vector<double>>& acc,
//     const thrust::device_vector<thrust::device_vector<thrust::device_vector<double>>>& rel_pos,
//     const thrust::device_vector<double>& density,
//     const double& rest_density,
//     const double& C_gas,
//     const double& C_h, const double& C_h6
// );

// __global__
// void cal_viscosity (
//     thrust::device_vector<thrust::device_vector<double>>& acc,
//     const thrust::device_vector<thrust::device_vector<thrust::device_vector<double>>>& rel_pos,
//     const thrust::device_vector<double>& density,
//     thrust::device_vector<thrust::device_vector<double>>& vel,
//     const double& mu,
//     const double& C_h, const double& C_h6
// );