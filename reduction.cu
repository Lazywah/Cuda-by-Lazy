#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <numeric>

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
    for (int i = 0; i < time; i++)
    {
        std::cout << "\033[0;30;43mCase #" << i + 1 << ":\033[0m\n";

        auto startMal = std::chrono::high_resolution_clock::now(); // start timer

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

        auto stopMal = std::chrono::high_resolution_clock::now();
        auto durationMal = std::chrono::duration_cast<std::chrono::microseconds>(stopMal - startMal).count() / 1000.0; // duration in milliseconds with three decimal points

        auto startH2D = std::chrono::high_resolution_clock::now(); // start timer

        // Copying our data onto the device (GPU)
        cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

        auto stopH2D = std::chrono::high_resolution_clock::now();
        auto durationH2D = std::chrono::duration_cast<std::chrono::microseconds>(stopH2D - startH2D).count() / 1000.0; // duration in milliseconds with three decimal points

        int blockSize = 256; // number of threads per block

        auto startRun = std::chrono::high_resolution_clock::now(); // start timer

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
        else
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

        auto stopRun = std::chrono::high_resolution_clock::now();
        auto durationRun = std::chrono::duration_cast<std::chrono::microseconds>(stopRun - startRun).count() / 1000.0; // duration in milliseconds with three decimal points

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
        std::cout << "\033[0;30;47mRun Time elapsed: " << durationRun << " ms\033[0m" << std::endl;
        std::cout << "D2H Time elapsed: " << durationD2H << " ms" << std::endl;
        std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << std::endl;

        // Freeing memory
        cudaFree(dev_input_data);
        cudaFree(dev_output_data);
        delete[] host_input_data;
        delete[] host_output_data;
    }
}

// I hope to use this main file for all of the reduction files
int main() {
Choice:
    int version, time;
    std::cout << "Enter the running Version(1~7) or Run All Verion(8): ";
    std::cin >> version;
    std::cout << "Enter the number of runs: ";
    std::cin >> time;
    std::cout << "\n";
    if (version < 1 || version>8)
    {
        std::cout << "Out if range! Please choice again!";
        goto Choice;
    }
    runversionwithoutoutput();
    if (version != 8)
        runversion(version, time);
    else
        for (int ver = 1; ver < 8; ver++)
        {
            std::cout << "\033[33mVERSION #" << ver << ":\033[0m\n\n";
            runversion(ver, time);
            std::cout << "\n\n";
        }
}