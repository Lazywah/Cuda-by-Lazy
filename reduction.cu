#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>  
#include <filesystem>
#include <windows.h>    // For CreateDirectoryA (Windows API)
#include <limits>       // For std::numeric_limits

#define Mal 0
#define H2D 1
#define Run 2
#define D2H 3
#define BW 4

#define For 1
#define ForBW 2

//[version][type][times]
double *allTime[7][5];
//[Mal][Rdc][RdcBW]
double *cpuTime[3];
std::string title[5] = {"Time_Mal(ms)", "Time_H2D(ms)", "Time_Run(ms)", "Time_D2H(ms)", "BandWidth(GB/s)"};

// REDUCTION 0 – Interleaved Addressing
__global__ void reduce0(int* g_in_data, int* g_out_data) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// REDUCTION 1 – Interleaved Addressing without branch divergence
__global__ void reduce1(int* g_in_data, int* g_out_data) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // note the stride as s *= 2 : this causes the interleaving addressing
        int index = 2 * s * tid;    // now we don't need a diverging branch from the if condition
        if (index + s < blockDim.x)
        {
            sdata[index] += sdata[index + s];   // s is used to denote the offset that will be combined
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// REDUCTION 2 – Sequence Addressing
__global__ void reduce2(int* g_in_data, int* g_out_data) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        // check out the reverse loop above
        if (tid < s) {   // then, we check threadID to do our computation
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// REDUCTION 3 – First Add During Load
__global__ void reduce3(int* g_in_data, int* g_out_data) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        // check out the reverse loop above
        if (tid < s) {   // then, we check tid to do our computation
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// Adding this function to help with unrolling
__device__ void warpReduce4(volatile int* sdata, int tid) {
    // the aim is to save all the warps from useless work 
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// REDUCTION 4 – Unroll Last Warp
__global__ void reduce4(int* g_in_data, int* g_out_data) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {  // only changing the end limit
        // check out the reverse loop above
        if (tid < s) {   // then, we check tid to do our computation
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Adding this to use warpReduce4
    if (tid < 32) {
        warpReduce4(sdata, tid);
    }

    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// Adding this function to help with unrolling and adding the Template
template <unsigned int blockSize>
__device__ void warpReduce5(volatile int* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// REDUCTION 5 – Completely Unroll
template <unsigned int blockSize>
__global__ void reduce5(int* g_in_data, int* g_out_data) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    // Perform reductions in steps, reducing thread synchronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) warpReduce5<blockSize>(sdata, tid);

    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// Adding this function to help with unrolling and adding the Template
template <unsigned int blockSize>
__device__ void warpReduce6(volatile int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// REDUCTION 6 – Multiple Adds / Threads
template <int blockSize>
__global__ void reduce6(int* g_in_data, int* g_out_data, unsigned int n) {
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_in_data[i] + g_in_data[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    // Perform reductions in steps, reducing thread synchronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) warpReduce6<blockSize>(sdata, tid);

    if (tid == 0) {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

void runversionwithoutoutput(int version=7)
{
    int n = 1 << 22; // Increase to about 4M elements
    size_t bytes = n * sizeof(int);

    // Host/CPU arrays
    int* host_input_data = new int[n];
    int* host_output_data = new int[(n + 255) / 256]; // to have sufficient size for output array

    // Device/GPU arrays
    int* dev_input_data, * dev_output_data;

    // Init data
    srand(42); // Fixed seed
    for (int i = 0; i < n; i++) {
        host_input_data[i] = rand() % 100;
    }

    // Allocating memory on GPU for device arrays
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_output_data, (n + 255) / 256 * sizeof(int));

    // Copying our data onto the device (GPU)
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256; // number of threads per block

    // Launch Kernel and Synchronize threads
    int num_blocks;

    num_blocks = (n + (2 * blockSize) - 1) / (2 * blockSize);   // Modifying this to account for the fact that 1 thread accesses 2 elements

    switch (blockSize) {
    case 512:
        reduce6<512> << <num_blocks, 512, 512 * sizeof(int) >> > (dev_input_data, dev_output_data, n);
        break;
    case 256:
        reduce6<256> << <num_blocks, 256, 256 * sizeof(int) >> > (dev_input_data, dev_output_data, n);
        break;
    case 128:
        reduce6<128> << <num_blocks, 128, 128 * sizeof(int) >> > (dev_input_data, dev_output_data, n);
        break;
    }
    cudaDeviceSynchronize();

    // Copying data back to the host (CPU)
    cudaMemcpy(host_output_data, dev_output_data, (n + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    int finalResult = host_output_data[0];
    for (int i = 1; i < (n + 255) / 256; ++i) {
        finalResult += host_output_data[i];
    }

    // Freeing memory
    cudaFree(dev_input_data);
    cudaFree(dev_output_data);
    delete[] host_input_data;
    delete[] host_output_data;
}

void runversion(int version, int time)
{
    for (int t = 0; t < time; t++)
    {
        std::cout << "\033[0;30;43mCase #" << t + 1 << ":\033[0m\n";

        int n = 1 << 22; // Increase to about 4M elements
        size_t bytes = n * sizeof(int);

        // Host/CPU arrays
        int* host_input_data = new int[n];
        int* host_output_data = new int[(n + 255) / 256]; // to have sufficient size for output array

        // Device/GPU arrays
        int* dev_input_data, * dev_output_data;

        // Init data
        srand(42); // Fixed seed
        for (int i = 0; i < n; i++) {
            host_input_data[i] = rand() % 100;
        }

        auto startMal = std::chrono::high_resolution_clock::now(); // start timer

        // Allocating memory on GPU for device arrays
        cudaMalloc(&dev_input_data, bytes);
        cudaMalloc(&dev_output_data, (n + 255) / 256 * sizeof(int));

        auto stopMal = std::chrono::high_resolution_clock::now();
        auto durationMal = std::chrono::duration_cast<std::chrono::microseconds>(stopMal - startMal).count() / 1000.0; // duration in milliseconds with three decimal points

        auto startH2D = std::chrono::high_resolution_clock::now(); // start timer

        // Copying our data onto the device (GPU)
        cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

        auto stopH2D = std::chrono::high_resolution_clock::now();
        auto durationH2D = std::chrono::duration_cast<std::chrono::microseconds>(stopH2D - startH2D).count() / 1000.0; // duration in milliseconds with three decimal points

        int blockSize = 256; // number of threads per block

        cudaEvent_t startRun, stopRun;
        cudaEventCreate(&startRun);
        cudaEventCreate(&stopRun);

        cudaEventRecord(startRun, 0);

        // Launch Kernel and Synchronize threads
        int num_blocks;

        if (version < 4)
        {
            num_blocks = (n + blockSize - 1) / blockSize;
        }
        else
        {
            num_blocks = (n + (2 * blockSize) - 1) / (2 * blockSize);   // Modifying this to account for the fact that 1 thread accesses 2 elements
        }
        cudaError_t err;

        if (version == 1)
            reduce0 << <num_blocks, blockSize, blockSize * sizeof(int) >> > (dev_input_data, dev_output_data);
        else if (version == 2)
            reduce1 << <num_blocks, blockSize, blockSize * sizeof(int) >> > (dev_input_data, dev_output_data);
        else if (version == 3)
            reduce2 << <num_blocks, blockSize, blockSize * sizeof(int) >> > (dev_input_data, dev_output_data);
        else if (version == 4)
            reduce3 << <num_blocks, blockSize, blockSize * sizeof(int) >> > (dev_input_data, dev_output_data);
        else if (version == 5)
            reduce4 << <num_blocks, blockSize, blockSize * sizeof(int) >> > (dev_input_data, dev_output_data);
        else if (version == 6)
        {
            // Needed for Complete unrolling
            switch (blockSize) {
            case 512:
                reduce5<512> << <num_blocks, 512, 512 * sizeof(int) >> > (dev_input_data, dev_output_data);
                break;
            case 256:
                reduce5<256> << <num_blocks, 256, 256 * sizeof(int) >> > (dev_input_data, dev_output_data);
                break;
            case 128:
                reduce5<128> << <num_blocks, 128, 128 * sizeof(int) >> > (dev_input_data, dev_output_data);
                break;
            }
        }
        else if(version == 7)
        {
            switch (blockSize) {
            case 512:
                reduce6<512> << <num_blocks, 512, 512 * sizeof(int) >> > (dev_input_data, dev_output_data, n);
                break;
            case 256:
                reduce6<256> << <num_blocks, 256, 256 * sizeof(int) >> > (dev_input_data, dev_output_data, n);
                break;
            case 128:
                reduce6<128> << <num_blocks, 128, 128 * sizeof(int) >> > (dev_input_data, dev_output_data, n);
                break;
            }
        }

        if (version < 6)
        {
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            }
        }
        cudaDeviceSynchronize();

        cudaEventRecord(stopRun, 0);
        cudaEventSynchronize(stopRun);

        float elapsedTime_ms = 0;
        cudaEventElapsedTime(&elapsedTime_ms, startRun, stopRun);
        double durationRun = (double)elapsedTime_ms;

        auto startD2H = std::chrono::high_resolution_clock::now(); // start timer

        // Copying data back to the host (CPU)
        cudaMemcpy(host_output_data, dev_output_data, (n + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

        auto stopD2H = std::chrono::high_resolution_clock::now();
        auto durationD2H = std::chrono::duration_cast<std::chrono::microseconds>(stopD2H - startD2H).count() / 1000.0; // duration in milliseconds with three decimal points

        // Final reduction on the host
        int finalResult = host_output_data[0];
        for (int i = 1; i < (n + 255) / 256; ++i) {
            finalResult += host_output_data[i];
        }

        // CPU Summation for verification
        int cpuResult = std::accumulate(host_input_data, host_input_data + n, 0);
        if (cpuResult == finalResult) {
            std::cout << "\033[32m"; // Set text color to green
            std::cout << "Verification successful: GPU result matches CPU result.\n";
            std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
        }
        else {
            std::cout << "\033[31m"; // Set text color to red
            std::cout << "Verification failed: GPU result (" << finalResult << ") does not match CPU result (" << cpuResult << ").\n";
            std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
        }
        std::cout << "\033[0m"; // Reset text color to default

        double bandwidth = (durationRun > 0) ? (bytes / durationRun / 1e6) : 0; // computed in GB/s, handling zero duration
        std::cout << "Reduced result: " << finalResult << std::endl;
        std::cout << "Malloc Time elapsed: " << durationMal << " ms" << std::endl;
        std::cout << "H2D Time elapsed: " << durationH2D << " ms" << std::endl;
        std::cout << "\033[0;30;47mRun Time elapsed: " << std::fixed << std::setprecision(3) << durationRun << " ms\033[0m" << std::endl;
        std::cout << "D2H Time elapsed: " << durationD2H << " ms" << std::endl;
        std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << std::endl;

        allTime[version - 1][Mal][t] = durationMal;
        allTime[version - 1][H2D][t] = durationH2D;
        allTime[version - 1][Run][t] = durationRun;
        allTime[version - 1][D2H][t] = durationD2H;
        allTime[version - 1][BW][t] = bandwidth;

        // Freeing memory
        cudaEventDestroy(startRun);
        cudaEventDestroy(stopRun);
        cudaFree(dev_input_data);
        cudaFree(dev_output_data);
        delete[] host_input_data;
        delete[] host_output_data;
    }
}

void runCPU(int time)
{
    for (int t = 0; t < time; t++)
    {
        std::cout << "\033[0;30;43mCase #" << t + 1 << ":\033[0m\n";

        int n = 1 << 22; // Increase to about 4M elements
        size_t bytes = n * sizeof(int);

        auto startMal = std::chrono::high_resolution_clock::now();

        // Host/CPU arrays
        int* host_input_data = new int[n];
        int* host_output_data = new int[(n + 255) / 256]; // to have sufficient size for output array

        auto stopMal = std::chrono::high_resolution_clock::now();
        auto durationMal = std::chrono::duration_cast<std::chrono::microseconds>(stopMal - startMal).count() / 1000.0; // duration in milliseconds with three decimal points

        // Init data
        srand(42); // Fixed seed
        for (int i = 0; i < n; i++) {
            host_input_data[i] = rand() % 100;
        }

        auto startRun = std::chrono::high_resolution_clock::now();

        int resultFor = 0;
        for (int i = 0; i < n; i++)
            resultFor += host_input_data[i];

        auto stopRun = std::chrono::high_resolution_clock::now();
        auto durationRun = std::chrono::duration_cast<std::chrono::microseconds>(stopRun - startRun).count() / 1000.0; // duration in milliseconds with three decimal points

        // CPU Summation for verification
        int cpuResult = std::accumulate(host_input_data, host_input_data + n, 0);
        if (cpuResult == resultFor) {
            std::cout << "\033[32m"; // Set text color to green
            std::cout << "Verification successful: GPU result matches CPU result.\n";
            std::cout << "GPU Result: " << resultFor << ", CPU Result: " << cpuResult << std::endl;
        }
        else {
            std::cout << "\033[31m"; // Set text color to red
            std::cout << "Verification failed: GPU result (" << resultFor << ") does not match CPU result (" << cpuResult << ").\n";
            std::cout << "GPU Result: " << resultFor << ", CPU Result: " << cpuResult << std::endl;
        }
        std::cout << "\033[0m"; // Reset text color to default

        double bandwidth = (durationRun > 0) ? (bytes / durationRun / 1e6) : 0; // computed in GB/s, handling zero duration
        std::cout << "Reduced result: " << resultFor << std::endl;
        std::cout << "Malloc Time elapsed: " << durationMal << " ms" << std::endl;
        std::cout << "\033[0;30;47mRun Time elapsed: " << durationRun << " ms\033[0m" << std::endl;
        std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << std::endl;

        cpuTime[Mal][t] = durationMal;
        cpuTime[For][t] = durationRun;
        cpuTime[ForBW][t] = bandwidth;

        delete[] host_input_data;
        delete[] host_output_data;
    }
}

std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    // Format: YYYY-MM-DD_HH-MM-SS
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void ExportDataWindows(int v, int t, const std::string& baseDirectory) {
    std::string folderName = "Data_" + GetTimestamp();
    std::string fullPath = baseDirectory + "\\" + folderName;

    if (CreateDirectoryA(fullPath.c_str(), NULL) ||
        GetLastError() == ERROR_ALREADY_EXISTS) {

        std::cout << "Successfully created new directory: " << fullPath << std::endl;
        
        std::string csvPath = fullPath + "\\Summary_Data.csv";
        std::string txtPath = fullPath + "\\Summary_Data.txt";

        std::ofstream csvFile(csvPath);
        if (csvFile.is_open()) {
            for (int tt = 0; tt < 5; tt++)
            {
                csvFile << title[tt];
                for (int i = 0; i < t; i++)
                    csvFile << ",test" << i + 1;
                csvFile << "\n";

                if (v == 9)
                {
                    for (int ver = 1; ver < 8; ver++)
                    {
                        csvFile << "Ver." << ver;
                        for (int test = 0; test < t; test++)
                            csvFile << "," << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << allTime[ver - 1][tt][test];
                        csvFile << "\n";
                    }
                    csvFile << "CPU";
                    if (tt == 0 || tt == 2 || tt == 4) 
                        for (int test = 0; test < t; test++)
                            csvFile << "," << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << cpuTime[tt/2][test];
                    csvFile << "\n\n";
                }
                else if(v != 8)
                {
                    csvFile << "Ver." << v;
                    for (int test = 0; test < t; test++)
                        csvFile << "," << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << allTime[v - 1][tt][test];
                    csvFile << "\n\n";
                }
                else
                {
                    csvFile << "CPU";
                    if (tt == 0 || tt == 2 || tt == 4)
                        for (int test = 0; test < t; test++)
                            csvFile << "," << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << cpuTime[tt / 2][test];
                    csvFile << "\n\n";
                }
            }

            csvFile.close();
            std::cout << "CSV file exported to directory.\n";
        }
        else {
            std::cerr << "\033[0;30;41mError: Could not create CSV file in directory.\033[0m\n";
        }

        std::ofstream txtFile(txtPath);
        if (txtFile.is_open()) {
            for (int tt = 0; tt < 5; tt++)
            {
                txtFile << title[tt];
                for (int i = 0; i < t; i++)
                    txtFile << " | test" << i + 1;
                txtFile << "\n";
                txtFile << "--";
                for (int i = 0; i < t; i++)
                    txtFile << "|--";
                txtFile << "\n";

                if (v == 9)
                {
                    for (int ver = 1; ver < 8; ver++)
                    {
                        txtFile << "Ver." << ver;
                        for (int test = 0; test < t; test++)
                            txtFile << " | " << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << allTime[ver - 1][tt][test];
                        txtFile << "\n";
                    }
                    if (tt == 0 || tt == 2 || tt == 4)
                    {
                        txtFile << "CPU";
                        for (int test = 0; test < t; test++)
                            txtFile << " | " << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << cpuTime[tt / 2][test];
                    }
                    txtFile << "\n\n";
                }
                else if(v != 8)
                {
                    txtFile << "Ver." << v;
                    for (int test = 0; test < t; test++)
                        txtFile << " | " << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << allTime[v - 1][tt][test];
                    txtFile << "\n\n";
                }
                else
                {
                    if (tt == 0 || tt == 2 || tt == 4)
                    {
                        txtFile << "CPU";
                        for (int test = 0; test < t; test++)
                            txtFile << " | " << std::fixed << std::setprecision(tt == 4 ? 4 : 3) << cpuTime[tt / 2][test];
                    }
                    txtFile << "\n\n";
                }
            }
            txtFile.close();
            std::cout << "TXT file exported to directory.\n";
        }
        else {
            std::cerr << "\033[0;30;41mError: Could not create TXT file in directory.\033[0m\n";
        }
        std::cout << "\033[0;30;42mFile exported successfully\033[0m\n\n";
    }
    else {
        std::cerr << "\033[0;30;41mError: Could not create directory " << fullPath << ". Error message: " << GetLastError() << "\033[0m\n";
    }
}

// I hope to use this main file for all of the reduction files
int main() {
Choice:
    int version, time;
    std::string YorN;
    std::cout << "Enter the running Version(1~7), CPU(8) or Run All Verion(9): ";
    std::cin >> version;
    std::cout << "Enter the number of runs: ";
    std::cin >> time;
    std::cout << "\n";
    if (version < 1 || version>9)
    {
        std::cout << "Out if range! Please choice again!";
        goto Choice;
    }

    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 5; j++)
            allTime[i][j] = new double[time];
    for (int i = 0; i < 3; i++)
        cpuTime[i] = new double[time];

    runversionwithoutoutput();
    if (version == 8)
    {
        std::cout << "\033[33mCPU" << ":\033[0m\n\n";
        runCPU(time);
        std::cout << "\n\n";
    }
    else if (version != 9)
    {
        std::cout << "\033[33mVERSION #" << version << ":\033[0m\n\n";
        runversion(version, time);
        std::cout << "\n\n";
    }
    else
    {
        for (int ver = 1; ver < 8; ver++)
        {
            std::cout << "\033[33mVERSION #" << ver << ":\033[0m\n\n";
            runversion(ver, time);
            std::cout << "\n\n";
        }
        std::cout << "\033[33mCPU" << ":\033[0m\n\n";
        runCPU(time);
        std::cout << "\n\n";
    }
YorN:
    std::cout << "\033[0;30;43mDo you want to export the data? [Y/N]:\033[0m ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    getline(std::cin, YorN);
    std::cout << "\n";
    if (YorN[0] == 'Y' || YorN[0] == 'y')
    {
        std::string exportBaseDir = "..\\data_resuction";
        if (CreateDirectoryA(exportBaseDir.c_str(), NULL) ||
            GetLastError() == ERROR_ALREADY_EXISTS) {
            ExportDataWindows(version, time, exportBaseDir);
        }
        else {
            std::cerr << "\033[0;30;41mError: Could not create directory " << exportBaseDir << ". Error message: " << GetLastError() << "\033[0m\n";
        }
    }
    else if (YorN[0] == 'N' || YorN[0] == 'n')
    {}
    else
    {
        std::cout << "\033[0;30;41mUnrecognized... Please enter again!\033[0m\n\n";
        goto YorN;
    }
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 5; j++)
            delete[] allTime[i][j];
    for (int i = 0; i < 3; i++)
        delete[] cpuTime[i];
}