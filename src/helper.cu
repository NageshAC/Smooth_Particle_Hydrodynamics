#include "helper.cuh"

   
void gpu_info(std::shared_ptr<cudaDeviceProp> props) {

    cudaGetDeviceProperties(props.get(), 0);


    #ifdef DEBUG
        static const int kb {1024};
        static const int mb {kb * kb};

        std::cout << "----------------------------GPU info----------------------------" << std::endl << std::endl;

        std::cout << "CUDA version:\t\tv" << CUDART_VERSION << std::endl << std::endl << std::endl;

        std::wcout << "CUDA Devices: " << std::endl << std::endl;

        
        std::wcout << "Device " << 0 << ": " << props->name << ": " << props->major << "." << props->minor << std::endl << std::endl;
        std::wcout << "       Global memory:   " << props->totalGlobalMem / mb << "mb" << std::endl;
        std::wcout << "       Shared memory:   " << props->sharedMemPerBlock / kb << "kb" << std::endl;
        std::wcout << "       Constant memory: " << props->totalConstMem / kb << "kb" << std::endl;
        std::wcout << "       Block registers: " << props->regsPerBlock << std::endl << std::endl;

        std::wcout << "       Warp size:             " << props->warpSize << std::endl;
        std::wcout << "       Max Threads per block: " << props->maxThreadsPerBlock << std::endl;
        std::wcout << "       Max block dimensions:  [ " << props->maxThreadsDim[0] << ", " << props->maxThreadsDim[1]  << ", " << props->maxThreadsDim[2] << " ]" << std::endl;
        std::wcout << "       Max grid dimensions:   [ " << props->maxGridSize[0] << ", " << props->maxGridSize[1]  << ", " << props->maxGridSize[2] << " ]" << std::endl;
        std::wcout << std::endl;

        std::cout << "----------------------------------------------------------------" << std::endl;
    #endif

}

unsigned _rel_count (const unsigned & n) { return (n * (n-1)) / 2; }

__host__
dim3 _get_block_size (
    const unsigned & n, 
    std::shared_ptr<cudaDeviceProp> props
){
    if (n <= props->maxThreadsPerBlock)
        return dim3 ();

    if (unsigned(n/props->maxThreadsPerBlock) < props->maxThreadsDim[2])
        return dim3(1, 1, unsigned(n/props->maxThreadsPerBlock) + 1);

    return dim3 ( 1,
        unsigned(n/props->maxThreadsPerBlock/props->maxThreadsDim[2]) + 1, 
        unsigned(props->maxThreadsDim[2]));
}

__host__
unsigned _get_thread_size (
    const unsigned & n, 
    std::shared_ptr<cudaDeviceProp> props
){
    if (n < props->warpSize)
        return props->warpSize;
    if (n < props->maxThreadsPerBlock)
        return n;

    return props->maxThreadsPerBlock;
}

__global__
void cal_rel_idx (
    unsigned * const rel_idx,
    const unsigned & N
){

    /* 
        idx / i --------------->
                 0  1  2  3  4

            0    x  x  x  x  x
            1    0  x  x  x  x
            2    1  2  x  x  x
            3    3  4  5  x  x
            4    6  7  8  9  x
    */
    const unsigned idx {blockIdx.x * blockDim.x + threadIdx.x}; // calculate core idx
    if (idx >= N) return; // safety check
    if (idx == 0) return; // just filling lower triangle with null diagonal

    const unsigned first {_rel_count(idx)};

    for (int i = 0; i < idx; i++) {
        rel_idx[(first + i) * depth_rel_idx]       = idx;
        rel_idx[(first + i) * depth_rel_idx + 1]   = i;
    }
}

// __device__ __host__ 
// inline unsigned _idx3D (
//     const unsigned x, 
//     const unsigned & dimY, 
//     const unsigned y, 
//     const unsigned & dimZ, 
//     const unsigned z
// ){
//     return (x * dimY + y )* dimZ + z ;
// }

// __device__ __host__ 
// inline unsigned _idx2D (
//     const unsigned x, 
//     const unsigned y
// ){
//     //TODO: y < x
//     return _rel_count(x) + y ;
// }