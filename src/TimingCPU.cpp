/**************/
/* TIMING CPU */
/**************/

#include "TimingCPU.h"

#ifdef __linux__

    #include <sys/time.h>
    #include <stdio.h>

    TimingCPU::TimingCPU(): cur_time_(0) { StartCounter(); }

    TimingCPU::~TimingCPU() { }

    void TimingCPU::StartCounter()
    {
        struct timeval time;
        if(gettimeofday( &time, 0 )) return;
        cur_time_ = 1000000 * time.tv_sec + time.tv_usec;
    }

    double TimingCPU::GetCounter()
    {
        struct timeval time;
        if(gettimeofday( &time, 0 )) return -1;

        long cur_time = 1000000 * time.tv_sec + time.tv_usec;
        double sec = (cur_time - cur_time_) / 1000000.0;
        if(sec < 0) sec += 86400;
        cur_time_ = cur_time;

        return 1000.*sec;
    }

#elif _WIN32 || _WIN64
    #include <windows.h>
    #include <iostream>

    struct PrivateTimingCPU {
        double  PCFreq;
        __int64 CounterStart;
    };

    // --- Default constructor
    TimingCPU::TimingCPU() { privateTimingCPU = new PrivateTimingCPU; (*privateTimingCPU).PCFreq = 0.0; (*privateTimingCPU).CounterStart = 0; }

    // --- Default destructor
    TimingCPU::~TimingCPU() { }

    // --- Starts the timing
    void TimingCPU::StartCounter()
    {
        LARGE_INTEGER li;
        if(!QueryPerformanceFrequency(&li)) std::cout << "QueryPerformanceFrequency failed!\n";

        (*privateTimingCPU).PCFreq = double(li.QuadPart)/1000.0;

        QueryPerformanceCounter(&li);
        (*privateTimingCPU).CounterStart = li.QuadPart;
    }

    // --- Gets the timing counter in ms
    double TimingCPU::GetCounter()
    {
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return double(li.QuadPart-(*privateTimingCPU).CounterStart)/(*privateTimingCPU).PCFreq;
    }
#endif