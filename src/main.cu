#include <iostream>
#include <fstream>
#include <memory>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>


#include "smoothing_kernels.cuh"
#include "force_kernels.cuh"
#include "helper.cuh"

#define DEBUG

#include "TimingCPU.h"
#include "TimingGPU.cuh"

#define file_path "./data/init_setup.txt"


// declearation of variables
__device__ __managed__ unsigned m_N {},  m_N_rel{};
__device__ __managed__ float m_mass {}, m_C_h {};
thrust::device_vector<float> d_pos {};
// calculate differnte versions of m_C_h
// TODO: calculate differnte versions of m_C_h 

int main () {

    TimingGPU timer_GPU;

    std::shared_ptr<cudaDeviceProp> GPUprops {std::make_shared<cudaDeviceProp>()};
    gpu_info(GPUprops);

    

    // get setup parameters
        std::ifstream file (file_path);
        
        // FILE existance check
        if(!file) { 
            std::cout << "The file: " << file_path << " cannot be opened" << std::endl;
            exit(-1);
        }

        std::string line {};
        while (file >> line) {

            if (line == "mass") {
                file >> m_mass;
                #ifdef DEBUG
                    std::cout << "m_mass = " << m_mass << std::endl;
                #endif
                continue;
            }


            if (line == "scope_length") {
                file >> m_C_h;
                #ifdef DEBUG
                    std::cout << "m_C_h = " << m_C_h << std::endl;
                #endif
                continue;
            }

            if (line == "position") {
                file >> m_N;
                #ifdef DEBUG
                    std::cout << "m_N = " << m_N << std::endl;
                #endif
                d_pos.reserve(m_N);
                float temp {};
                for (int i = 0; i < m_N; i++) {
                    file >> temp;
                    d_pos.push_back(temp);
                }
            }

        }

        // #ifdef DEBUG
        //     std::copy(d_pos.begin(), d_pos.end(), std::ostream_iterator<float>(std::cout, " "));
        //     std::cout << std::endl;
        // #endif

        file.close();
    // end of parameters

    // declearation of variables
    m_N_rel = _rel_count(m_N);
    thrust::device_vector<float>        d_rel_pos (m_N_rel * depth_rel_pos, 0.f),
                                        d_density (m_N, 0.f);
    thrust::device_vector<unsigned>     d_rel_idx (m_N_rel * depth_rel_idx, 0);

    const dim3 blocks       {_get_block_size  (m_N, GPUprops)};
    const unsigned thread   {_get_thread_size (m_N, GPUprops)};
    
    printf ("Blocks = {%d, %d, %d}\nThreads = %d\n", blocks.x, blocks.y, blocks.z, thread);

    timer_GPU.StartCounter();
        cal_rel_idx <<<blocks,thread>>> (raw_pointer_cast(&d_rel_idx[0]), m_N);
    std::cout << "GPU Timing for cal_rel_idx = " << timer_GPU.GetCounter() << " ms for " << m_N << " elements" << std::endl;

    // #ifdef DEBUG
    //     std::copy(d_rel_idx.begin(), d_rel_idx.end(), std::ostream_iterator<unsigned>(std::cout, " "));
    //     std::cout << std::endl;
    // #endif



}
