#include "cuda_runtime.h"

extern "C" {
    __host__ __device__
    double add ( double& a, double& b) {
        return a + b;
    }

    // __host__ __device__
    // double add ( double* a,  double* b) {
    //     return add_ref(*a, *b);;
    // }

}