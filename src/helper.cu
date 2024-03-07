#include "helper.cuh"

   
void gpu_info(std::shared_ptr<cudaDeviceProp> props) {

    cudaGetDeviceProperties(props.get(), 0);


    // #ifdef DEBUG
    //     static const int kb {1024};
    //     static const int mb {kb * kb};

    //     std::cout << "----------------------------GPU info----------------------------" << std::endl << std::endl;

    //     std::cout << "CUDA version:\t\tv" << CUDART_VERSION << std::endl << std::endl << std::endl;

    //     std::wcout << "CUDA Devices: " << std::endl << std::endl;

        
    //     std::wcout << "Device " << 0 << ": " << props->name << ": " << props->major << "." << props->minor << std::endl << std::endl;
    //     std::wcout << "       Global memory:                " << props->totalGlobalMem / mb << "mb" << std::endl;
    //     std::wcout << "       Shared memory:                " << props->sharedMemPerBlock / kb << "kb" << std::endl;
    //     std::wcout << "       Constant memory:              " << props->totalConstMem / kb << "kb" << std::endl;
    //     std::wcout << "       Block registers:              " << props->regsPerBlock << std::endl;
    //     std::wcout << "       Num of multiprocessors (SM):  " << props->multiProcessorCount << std::endl << std::endl;

    //     std::wcout << "       Warp size:             " << props->warpSize << std::endl;
    //     std::wcout << "       Max Threads per block: " << props->maxThreadsPerBlock << std::endl;
    //     std::wcout << "       Max block dimensions:  [ " << props->maxThreadsDim[0] << ", " << props->maxThreadsDim[1]  << ", " << props->maxThreadsDim[2] << " ]" << std::endl;
    //     std::wcout << "       Max grid dimensions:   [ " << props->maxGridSize[0] << ", " << props->maxGridSize[1]  << ", " << props->maxGridSize[2] << " ]" << std::endl;
    //     std::wcout << std::endl;

    //     std::cout << "----------------------------------------------------------------" << std::endl;
    // #endif

}

// count of relative elements
unsigned _rel_count (const unsigned n) { return (n * (n-1)) / 2; }

// void _rec_idx (unsigned idx, unsigned& a, unsigned& b, const unsigned lvl){
//     if (lvl > idx) {
//         b = lvl;
//         a = idx;
//         return;
//     }

//     _rec_idx (idx - lvl, a, b, lvl+1);
// }

void _quad_idx (unsigned idx, unsigned& a, unsigned& b) {
    /* 
    * a x² + bx + c = 0
    * (-b ± √(b²-4ac)) / (2a)

    * This case
    *   n² + (-1)x + (-2 * idx) = 0
    *   (1 ± √(1 + 8 * idx))) / 2
    *   we need only positive number 
    *   unsigned ((1 + √(1 + 8 * idx)) / 2)
    */

    b = unsigned ((1 + sqrtf(1 + 8 * idx)) / 2);
    a = idx - _rel_count (b);
}

unsigned _rel_pos (const unsigned a, const unsigned b) {
    /*
    * a cannot be equal to b
    */
    if (b > a)
        return _rel_count(b) + a;
    return b + _rel_count(a);
}

__global__
void cal_rel_coords (
    Rel_Force_Vector<float>      * const __restrict__ f_vec,
    const Tensors<float> * const __restrict__ coords,
    const float     C_h, 
    const float     C_h2, 
    const float     C_1_h6, 
    const float     C_1_h9,
    const unsigned  rel_N
){
    const unsigned idx {blockIdx.x * blockDim.x + threadIdx.x}; // calculate core idx
    const unsigned stride {blockDim.x * gridDim.x};

    unsigned a{}, b{};
    for (int i = idx; i < rel_N; i += stride){
        _quad_idx (i, a, b);

        // calculate relative position
        f_vec[i].x = coords[b].x - coords[a].x;
        f_vec[i].y = coords[b].y - coords[a].y;
        f_vec[i].z = coords[b].z - coords[a].z;

        // calculate relative distance
        f_vec[i].dist = norm3df(f_vec[i].x, f_vec[i].y, f_vec[i].z);

        f_vec[i].dist_inv = 1 / f_vec[i].dist;

        // printf ("%d -> {%d, %d} => %f\n", i, b, a, f_vec[i].dist);

        // W_poly6 for density 
        f_vec[i].W_poly6.x = W_poly6(f_vec[i].x, C_h, C_h2, C_1_h9);
        f_vec[i].W_poly6.y = W_poly6(f_vec[i].y, C_h, C_h2, C_1_h9);
        f_vec[i].W_poly6.z = W_poly6(f_vec[i].z, C_h, C_h2, C_1_h9);

        // lap_W_poly6 surface tension 
        f_vec[i].lap_W_poly6.x = lap_W_poly6(f_vec[i].x, C_h, C_h2, C_1_h9);
        f_vec[i].lap_W_poly6.y = lap_W_poly6(f_vec[i].y, C_h, C_h2, C_1_h9);
        f_vec[i].lap_W_poly6.z = lap_W_poly6(f_vec[i].z, C_h, C_h2, C_1_h9);

        // grad_W_spiky for pressure
        f_vec[i].grad_W_spiky.x = grad_W_spiky(f_vec[i].x, C_h, C_1_h6);
        f_vec[i].grad_W_spiky.y = grad_W_spiky(f_vec[i].y, C_h, C_1_h6);
        f_vec[i].grad_W_spiky.z = grad_W_spiky(f_vec[i].z, C_h, C_1_h6);

        // lap_W_vis for Viscosity
        f_vec[i].lap_W_vis.x = lap_W_viscosity(f_vec[i].x, C_h, C_1_h6);
        f_vec[i].lap_W_vis.y = lap_W_viscosity(f_vec[i].y, C_h, C_1_h6);
        f_vec[i].lap_W_vis.z = lap_W_viscosity(f_vec[i].z, C_h, C_1_h6);
    }
}

/*
*   Smoothing Kernels
*/

// W_poly6 for density 
template <class Typ>
Typ W_poly6 (
    const Typ& r,
    const Typ& C_h,
    const Typ& C_h2,
    const Typ& C_1_h9
){
    if ((r) >= C_h) return 0.f;
    return Typ(C_POLY6 * C_1_h9 * pow(C_h2 - pow(r, 2.), 3.));
}

// lap_W_poly6 surface tension 
template <class Typ>
Typ lap_W_poly6 (
    const Typ& r,
    const Typ& C_h,
    const Typ& C_h2,
    const Typ& C_1_h9
){
    if (r >= C_h) return 0.;
    const Typ X {C_h2 - pow(r, 2.f)};
    return Typ(C_LAP_POLY6 * C_1_h9 * X * (4 * pow(r, 2.f) - X));
}

// grad_W_spiky for pressure
template <class Typ>
Typ grad_W_spiky (
    const Typ& r,
    const Typ& C_h,
    const Typ& C_1_h6
){
    if (r >= C_h) return 0.;

    return Typ(C_GRAD_SPIKY * C_1_h6 * pow(C_h - r, 2.f));
}

// lap_W_vis for Viscosity
template <class Typ>
Typ lap_W_viscosity (
    const Typ& r,
    const Typ& C_h,
    const Typ& C_1_h6
){
    if (r >= C_h) return 0.;

    return Typ(C_LAP_VISC * C_1_h6 * (C_h - r));
}