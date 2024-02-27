// 1 micro-second accuracy
// Returns the time in seconds

// Source: https://stackoverflow.com/questions/7876624/timing-cuda-operations

#ifndef __TIMINGCPU_H__
#define __TIMINGCPU_H__

#ifdef __linux__

    class TimingCPU {

        private:
            long cur_time_;

        public:

            TimingCPU();

            ~TimingCPU();

            void StartCounter();

            double GetCounter();
    };

#elif _WIN32 || _WIN64

    struct PrivateTimingCPU;

    class TimingCPU
    {
        private:
            PrivateTimingCPU *privateTimingCPU;

        public:

            TimingCPU();

            ~TimingCPU();

            void StartCounter();

            double GetCounter();

    }; // TimingCPU class

#endif

#endif