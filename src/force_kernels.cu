#include "force_kernels.cuh"

// __global__
// void cal_rel_pos (
//     thrust::device_vector<float>& rel_pos,
//     const thrust::device_vector<float>& pos
// ){
//     const unsigned idx {blockIdx.x * blockDim.x + threadIdx.x}; // calculate core idx
//     if (!(idx < rel_pos.size())) return; // safety check

    

    // if (idx == 0) return; // just filling lower triangle with null diagonal

    // for (int j = idx-1; j >= 0; j--) {
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 0)] = pos[_idx2D(j, dimX, 0)] - pos[_idx2D(idx, dimX, 0)];
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 1)] = pos[_idx2D(j, dimX, 1)] - pos[_idx2D(idx, dimX, 1)];
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 2)] = pos[_idx2D(j, dimX, 2)] - pos[_idx2D(idx, dimX, 2)];
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 3)] 
    //         =     std::sqrt(std::pow(rel_pos[_idx3D(idx, dimX, j, dimX, 0)],2) 
    //                       + std::pow(rel_pos[_idx3D(idx, dimX, j, dimX, 1)],2) 
    //                       + std::pow(rel_pos[_idx3D(idx, dimX, j, dimX, 2)],2));

    //     rel_pos[_idx3D(idx, dimX, j, dimX, 0)] = rel_pos[_idx3D(j, dimX, idx, dimX, 0)];
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 1)] = rel_pos[_idx3D(j, dimX, idx, dimX, 1)];
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 2)] = rel_pos[_idx3D(j, dimX, idx, dimX, 2)];
    //     rel_pos[_idx3D(idx, dimX, j, dimX, 3)] = rel_pos[_idx3D(j, dimX, idx, dimX, 3)];
    // }
// }

// void cal_density (
//     thrust::device_vector<double>& density,
//     thrust::device_vector<double>& C_1_rho,
//     const double& mass,
//     const thrust::device_vector<thrust::device_vector<thrust::device_vector<double>>>& rel_pos,
//     const double& C_h2, const double& C_h9
// ) {
//     const unsigned idx {blockIdx.x * blockDim.x + threadIdx.x};
//     if (!(idx < rel_pos.size())) return; // safety check

//     density[idx] = 0.; // reset density vector
//     for ( auto& rp : rel_pos [idx] ) {
//         const double r2 {std::pow(rp[3],2)};
//         density[idx] += W_poly6(r2, C_h2, C_h9);
//     }
//     density[idx] *= mass;
//     C_1_rho[idx]  = 1 / density[idx];
// }

// void cal_pressure (
//     thrust::device_vector<thrust::device_vector<double>>& acc,
//     const thrust::device_vector<thrust::device_vector<thrust::device_vector<double>>>& rel_pos,
//     const thrust::device_vector<double>& density,
//     const double& rest_density, const double& C_gas,
//     const double& C_h, const double& C_h6
// ) {
//     const unsigned idx {blockIdx.x * blockDim.x + threadIdx.x};
//     if (!(idx < rel_pos.size())) return; // safety check

//     for (unsigned j = 0; j < rel_pos.size(); ++j) {
//         if (idx == j) continue; // pressure forces due to itself
//         if (rel_pos[idx][j][3] == 0) continue; // division by 0 case

//         const double grad_spiky {grad_W_spiky(rel_pos[idx][j][3], C_h, C_h6)};
//         if (! grad_spiky) continue; // outside the neighborhood

//         unsigned double C {1};
//         C *= C_gas * grad_spiky;
//         C *= (density[idx] + density[j] - 2 * reset_density);
//         C /= 2 * density[j] * rel_pos[idx][j][3];

//         acc[idx][0] -= C * rel_pos[idx][j][0];
//         acc[idx][1] -= C * rel_pos[idx][j][1];
//         acc[idx][2] -= C * rel_pos[idx][j][2];
//     }
// }

// void cal_viscosity (
//     thrust::device_vector<thrust::device_vector<double>>& acc,
//     const thrust::device_vector<thrust::device_vector<thrust::device_vector<double>>>& rel_pos,
//     const thrust::device_vector<double>& C_1_rho,
//     thrust::device_vector<thrust::device_vector<double>>& vel,
//     const double& mu, const double& C_h, const double& C_h6
// ) {
//     const unsigned idx {blockIdx.x * blockDim.x + threadIdx.x};
//     if (!(idx < rel_pos.size())) return; // safety check

//     thrust::device_vector<double> acc_visc {3, 0.}

//     for (unsigned j = 0; j < rel_pos.size(); ++j) {
//         if (idx == j) continue; // viscous forces due to itself
//         unsigned double C {1};
//         C *= lap_W_viscosity(rel_pos[idx][j][3], C_h, C_h6);
//         if (!C) continue; // outside the neighborhood
//         C *= C_1_rho[j];

//         acc_visc[0] += C * (vel[j][0] - vel[idx][0]);
//         acc_visc[1] += C * (vel[j][1] - vel[idx][1]);
//         acc_visc[2] += C * (vel[j][2] - vel[idx][2]);
//     }
//     acc[idx][0] += mu * cc_visc[0];
//     acc[idx][1] += mu * cc_visc[1];
//     acc[idx][2] += mu * cc_visc[2];
// }