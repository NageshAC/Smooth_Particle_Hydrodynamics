#include <iostream>
#include <fstream>
#include <memory>
#include <assert.h>

#define DEBUG

#include "helper.cuh"
#include "class.cuh"

#include "TimingCPU.h"
#include "TimingGPU.cuh"

typedef unsigned long long ULL;

#define file_path "./data/init_setup.txt"

cudaError_t checkCuda(cudaError_t);

int main () {
    
    // declearation of variables
    std::shared_ptr<cudaDeviceProp> GPUprops {std::make_shared<cudaDeviceProp>()};
    gpu_info(GPUprops);
    static const unsigned  blocks {80}, 
    // static const unsigned  blocks {unsigned(GPUprops -> multiProcessorCount)}, 
                    ThreadsPerBlock {unsigned(GPUprops -> maxThreadsPerBlock)};
    
    TimingGPU timer_GPU;

    unsigned  C_N {};
    float     C_mass {}, C_h {};

    // deleters for smart pointers
    auto cuda_deleter_Tensors = [&] (Tensors<float>* ptr) {cudaFree(ptr);};
    auto cuda_deleter_Rel_Force_Vector = [&] (Rel_Force_Vector<float>* ptr) {cudaFree(ptr);};

    std::shared_ptr<Tensors<float>[]>           h_pos   {nullptr}, 
                                                d_pos   {nullptr, cuda_deleter_Tensors};
    std::shared_ptr<Rel_Force_Vector<float>[]>  d_f_vec {nullptr, cuda_deleter_Rel_Force_Vector};

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
                file >> C_mass;
                #ifdef DEBUG
                    std::cout << "C_mass = " << C_mass << std::endl;
                #endif
                continue;
            }


            if (line == "scope_length") {
                file >> C_h;
                #ifdef DEBUG
                    std::cout << "C_h = " << C_h << std::endl;
                #endif
                continue;
            }

            if (line == "position") {
                file >> C_N;
                #ifdef DEBUG
                    std::cout << "C_N = " << C_N << std::endl;
                #endif
                h_pos = std::make_shared<Tensors<float>[]>(C_N);

                for ( int i = 0; i < C_N; i++) {
                    file >> h_pos[i].x >> h_pos[i].y >> h_pos[i].z;
                }
                
                continue;
            }

        }

        // #ifdef DEBUG
        //     std::cout << "h_pos.size() = " << C_N << "\n";
        //     for (int i = 0; i < C_N; i++) {
        //         std::cout   << std::setprecision(6) << std::fixed
        //                     << h_pos[i].x << "\t" 
        //                     << h_pos[i].y << "\t" 
        //                     << h_pos[i].z << std::endl;
        //     }
        // #endif

        file.close();
    // end of parameters

    // define all sizes
    static const ULL C_rel_N {_rel_count(C_N)},
                     size_pos {C_N * sizeof(Tensors<float>)},
                     size_f_vec {C_rel_N * sizeof(Rel_Force_Vector<float>)};
    printf("C_rel_N = %lld\n", C_rel_N);
    printf("Total size of Position Vector = %lld Bits = %lld bytes\n", size_pos, size_pos/8);
    printf("Total size of Force Vector = %lld Bits = %lld bytes\n", size_f_vec, size_f_vec/8);

    // TODO: calculate differnte versions of C_h 
    static const float  C_h2    {float(std::pow(C_h, 2))}, 
                        C_1_h6  {float(1 / std::pow(C_h, 6))}, 
                        C_1_h9  {float(1 / std::pow(C_h, 9))};
    
    timer_GPU.StartCounter();
    // allocate device memory for d_pos
    checkCuda(cudaMalloc((void **)&d_pos, size_pos));
    // copy positions: host -> device memory 
    checkCuda(cudaMemcpy(d_pos.get(), h_pos.get(), size_pos, cudaMemcpyHostToDevice));
    // allocate device memory for d_f_vec
    checkCuda(cudaMalloc((void **)&d_f_vec, size_f_vec));
    std::cout << "Allocation and Copy Time = " << timer_GPU.GetCounter() << " ms" << std::endl;


    timer_GPU.StartCounter();
    cal_rel_coords <<<blocks, ThreadsPerBlock>>> (
        d_f_vec.get(), 
        d_pos.get(), 
        C_h, C_h2, C_1_h6, C_1_h9, C_rel_N
    );
    cudaDeviceSynchronize();
    std::cout << "cal_rel_coords <<<"
              << blocks << ", " << ThreadsPerBlock
              << ">>> Time = " << timer_GPU.GetCounter() << " ms" 
              << std::endl;


}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}