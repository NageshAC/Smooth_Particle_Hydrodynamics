#include "../src/smoothing_kernels.cpp"

// Wrapper around functions for ctypes functions (python)
extern "C" {

    const double p_W_poly6 (
        const double& r2, 
        const double& C_h2, 
        const double& C_1_h9
    ) { return W_poly6(r2, C_h2, C_1_h9); }

    double p_grad_W_poly6 (
        const double& r, 
        const double& r2, 
        const double& C_h2, 
        const double& C_1_h9
    ) { return grad_W_poly6(r, r2, C_h2, C_1_h9); }

    double p_W_spiky (
        const double& r,
        const double& C_h,
        const double& C_1_h6
    ) { return W_spiky(r, C_h, C_1_h6); }

    double p_grad_W_spiky (
        const double& r,
        const double& C_h,
        const double& C_1_h6
    ) { return grad_W_spiky(r, C_h, C_1_h6); }

    double p_lap_W_viscosity (
        const double& r,
        const double& C_h,
        const double& C_1_h6
    ) { return lap_W_viscosity(r, C_h, C_1_h6); }

} // extern "C" namespace