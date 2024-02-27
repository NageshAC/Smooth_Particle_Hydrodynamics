#include "smoothing_kernels.cuh"

double W_poly6 (
    const double& r2, 
    const double& C_h2, 
    const double& C_1_h9
){
    if (r2 >= C_h2) return 0.;
    // std::cout << r2 << std::endl;
    return C_POLY6 * C_1_h9 * pow(C_h2 - r2, 3);
}

double grad_W_poly6 (
    const double& r, 
    const double& r2, 
    const double& C_h2, 
    const double& C_1_h9
){
    if (r2 >= C_h2) return 0.;
    return C_GRAD_POLY6 * C_1_h9 * r * pow(C_h2 - r2, 2);
}

double W_spiky (
    const double& r,
    const double& C_h,
    const double& C_1_h6
){
    if (r >= C_h) return 0.;
    return C_SPIKY * C_1_h6 * pow(C_h - r, 3);
}

double grad_W_spiky (
    const double& r,
    const double& C_h,
    const double& C_1_h6
){
    if (r >= C_h) return 0.;

    return C_GRAD_SPIKY * C_1_h6 * pow(C_h - r, 2);
}

double lap_W_viscosity (
    const double& r,
    const double& C_h,
    const double& C_1_h6
){
    if (r >= C_h) return 0.;

    return C_LAP_VISC * C_1_h6 * (C_h - r);
}