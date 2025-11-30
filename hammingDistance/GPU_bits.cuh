#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "struct.h"

using namespace std;

typedef unsigned long long ULL;
typedef unsigned char UC;

#define MAX_TEXT_LENGTH 10000000
#define CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

__global__ void encodeAndHamming(
    const char* text, UC* d_alphabet, UC* dist, int L, ULL lengthText, ULL ref, ULL clearWitnessBit, ULL keepWitnessBit, int threadtimes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bitPerSymbol = 3;
    int numWindows = lengthText - L + 1;
    int i, j, k;
    int pos = (tid)*threadtimes;

    if (tid * threadtimes >= numWindows) return;

    ULL code = 0;
    char ch;

    for (j = 0; j < L - 1; j++)
        code = (code << bitPerSymbol) | d_alphabet[text[pos + j]];

    for (i = 0, j = L - 1; i < threadtimes; i++, j++)
    {
        code = (code << bitPerSymbol) | d_alphabet[text[pos + j]];

        dist[pos + i] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);
    }
}

gpuData GPU_bits_hammingDistance(string inputFile, int L, int K, int* answer)
{
    clock_t readstart = clock();
    FILE* f = fopen(inputFile.c_str(), "rb");
    if (!f) 
    {
        printf("\033[0;37;41m❌ 無法開啟檔案: %s\033[0m\n", inputFile.c_str());
        return gpuData();
    }

    fseek(f, 0, SEEK_END);
    int lengthText = ftell(f);
    rewind(f);

    char* h_text = (char*)malloc(lengthText * sizeof(char));
    fread(h_text, sizeof(char), lengthText, f);
    fclose(f);
    clock_t readend = clock();

    int numWindows = lengthText - L + 1;

    UC alphabet[255] = { 0 };
    alphabet['A'] = 0;
    alphabet['C'] = 1;
    alphabet['G'] = 2;
    alphabet['T'] = 3;

    ULL ref = 0;
    int bitPerSymbol = 3;
    char ch;
    int i = 0, j = 0, k = 0;
    for (j = 0; j < L; j++) 
    {
        ch = h_text[j];
        ref = (ref << bitPerSymbol) | alphabet[ch];
    }

    ULL clearWitnessBit, keepWitnessBit;
    ULL filterUselessBit = ~0x0ULL;
    keepWitnessBit = clearWitnessBit = 0x1ULL << (bitPerSymbol - 1);

    for (k = 1; k < L; k++)
        keepWitnessBit = (keepWitnessBit << bitPerSymbol) | clearWitnessBit;
    clearWitnessBit = (~keepWitnessBit) & filterUselessBit;

    UC* h_dist = (UC*)malloc((numWindows + 100) * sizeof(UC));

    char* d_text;
    UC* d_dist;
    UC* d_alphabet;
    CHECK(cudaMalloc(&d_text, (lengthText + 100) * sizeof(char)));

    CHECK(cudaMalloc(&d_dist, (numWindows + 100) * sizeof(UC)));
    CHECK(cudaMalloc(&d_alphabet, 255 * sizeof(UC)));
    clock_t popcountstart, popcountend;
    clock_t h2dstart = clock();
    CHECK(cudaMemcpy(d_text, h_text, lengthText * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_alphabet, alphabet, 255 * sizeof(UC), cudaMemcpyHostToDevice));
    clock_t h2dend = clock();

    popcountstart = clock();
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int maxThreadsPerBlock = 1024;
    int threadtimes = 16;

    int blockSize = maxThreadsPerBlock / threadtimes;
    int totalThreads = (numWindows + threadtimes - 1) / threadtimes;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    encodeAndHamming << <gridSize, blockSize >> > (d_text, d_alphabet, d_dist, L, lengthText, ref, clearWitnessBit, keepWitnessBit, threadtimes);

    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    popcountend = clock();
    printf("GPU Hamming 計算完成，耗時 %.2f 毫秒\n", elapsed);

    clock_t d2hstart = clock();
    CHECK(cudaMemcpy(h_dist, d_dist, numWindows * sizeof(UC), cudaMemcpyDeviceToHost));
    clock_t d2hend = clock();

    double readtime = (double)(readend - readstart) / CLOCKS_PER_SEC;
    double h2d_t = (double)(h2dend - h2dstart) / CLOCKS_PER_SEC;
    double popcount_t = (double)(popcountend - popcountstart) / CLOCKS_PER_SEC;
    double d2h_t = (double)(d2hend - d2hstart) / CLOCKS_PER_SEC;
    //double total_t = (double)(d2hend - h2dstart) / CLOCKS_PER_SEC;
    double total_t = readtime + h2d_t + popcount_t + d2h_t;

    printf("\n");
    printf("inputFile: %s\n", inputFile.c_str());
    printf("readtime: %.6f \n", readtime);
    printf("h2d_time: %.6f \n", h2d_t);
    printf("popcount_t: %.6f \n", popcount_t);
    printf("d2h_time: %.6f \n", d2h_t);
    printf("total_time: %.6f \n", total_t);
    printf("\n");

    printf("dist[] = {");
    for (int i = 0; i < 30; i++)
        printf(" %d%s", h_dist[i], (i == 29 ? " " : ","));
    printf("...}\n");

    printf("ans[]  = {");
    for (int i = 0; i < 30; i++)
        printf(" %d%s", answer[i], (i == 29 ? " " : ","));
    printf("...}\n");

    for (int i = 0; i < numWindows; i++)
        if (h_dist[i] != answer[i])
        {
            cout << "\033[0;37;41m計算錯誤!\033[0m\n";
            cout << "dist 長度: " << numWindows << "\n";
            cout << "在 dist[" << i << "] 發現錯誤: r: " << h_dist[i] << " != a: " << answer[i] << "\n";
            cudaFree(d_text);
            cudaFree(d_dist);
            free(h_text);
            free(h_dist);
            return 1;
        }

    cout << "\033[0;37;42m計算正確\033[0m\n";
    cudaFree(d_text);
    cudaFree(d_dist);
    free(h_text);
    free(h_dist);

    return gpuData(readtime, h2d_t, popcount_t, d2h_t, total_t);
}

int bits_hammingDistance_noPrint(string inputFile, int L, int K)
{
    clock_t readstart = clock();
    FILE* f = fopen(inputFile.c_str(), "rb");
    if (!f) 
    {
        printf("❌ 無法開啟檔案: %s\n", inputFile.c_str());
        return 1;
    }

    fseek(f, 0, SEEK_END);
    int lengthText = ftell(f);
    rewind(f);

    char* h_text = (char*)malloc(lengthText * sizeof(char));
    fread(h_text, sizeof(char), lengthText, f);
    fclose(f);
    clock_t readend = clock();

    int numWindows = lengthText - L + 1;

    UC alphabet[255] = { 0 };
    alphabet['A'] = 0;
    alphabet['C'] = 1;
    alphabet['G'] = 2;
    alphabet['T'] = 3;

    ULL ref = 0;
    int bitPerSymbol = 3;
    char ch;
    int i = 0, j = 0, k = 0;
    for (j = 0; j < L; j++) 
    {
        ch = h_text[j];
        ref = (ref << bitPerSymbol) | alphabet[ch];
    }

    ULL clearWitnessBit, keepWitnessBit;
    ULL filterUselessBit = ~0x0ULL;
    keepWitnessBit = clearWitnessBit = 0x1ULL << (bitPerSymbol - 1);

    for (k = 1; k < L; k++) 
        keepWitnessBit = (keepWitnessBit << bitPerSymbol) | clearWitnessBit;
    clearWitnessBit = (~keepWitnessBit) & filterUselessBit;

    UC* h_dist = (UC*)malloc((numWindows + 100) * sizeof(UC));

    char* d_text;
    UC* d_dist;
    UC* d_alphabet;
    CHECK(cudaMalloc(&d_text, (lengthText + 100) * sizeof(char)));

    CHECK(cudaMalloc(&d_dist, (numWindows + 100) * sizeof(UC)));
    CHECK(cudaMalloc(&d_alphabet, 255 * sizeof(UC)));
    clock_t popcountstart, popcountend;
    clock_t h2dstart = clock();
    CHECK(cudaMemcpy(d_text, h_text, lengthText * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_alphabet, alphabet, 255 * sizeof(UC), cudaMemcpyHostToDevice));
    clock_t h2dend = clock();

    popcountstart = clock();
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int maxThreadsPerBlock = 1024;
    int threadtimes = 16;

    int blockSize = maxThreadsPerBlock / threadtimes;
    int totalThreads = (numWindows + threadtimes - 1) / threadtimes;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    encodeAndHamming << <gridSize, blockSize >> > (d_text, d_alphabet, d_dist, L, lengthText, ref, clearWitnessBit, keepWitnessBit, threadtimes);

    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    popcountend = clock();

    clock_t d2hstart = clock();
    CHECK(cudaMemcpy(h_dist, d_dist, numWindows * sizeof(UC), cudaMemcpyDeviceToHost));
    clock_t d2hend = clock();

    cudaFree(d_text);
    cudaFree(d_dist);
    free(h_text);
    free(h_dist);
}