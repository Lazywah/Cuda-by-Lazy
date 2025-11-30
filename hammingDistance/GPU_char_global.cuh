#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "struct.h"

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

__global__ void hammingDistanceGlobal(
    const char* d_text, const char* d_pattern, unsigned char* d_dist, int L, int lengthText)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int numWindows = lengthText - L + 1;

    if (i >= numWindows) return;

    int distance = 0;

    for (int j = 0; j < L; j++)
        if (d_text[i + j] != d_pattern[j])
            distance++;

    d_dist[i] = (unsigned char)distance;
}

gpuData GPU_char_hammingDistance_global(string inputFile, int L, int K, int* answer)
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

    char* h_text = (char*)malloc(lengthText + 1);
    fread(h_text, 1, lengthText, f);
    h_text[lengthText] = '\0';
    fclose(f);
    clock_t readend = clock();

    int numWindows = lengthText - L + 1;

    char* h_pattern = h_text;
    unsigned char* h_dist = (unsigned char*)malloc(numWindows * sizeof(unsigned char));

    char* d_text;
    char* d_pattern;
    unsigned char* d_dist;

    CHECK(cudaMalloc(&d_text, lengthText * sizeof(char)));
    CHECK(cudaMalloc(&d_pattern, L * sizeof(char)));
    CHECK(cudaMalloc(&d_dist, numWindows * sizeof(unsigned char)));
    clock_t popcountstart, popcountend;
    clock_t h2dstart = clock();
    CHECK(cudaMemcpy(d_text, h_text, lengthText * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pattern, h_pattern, L * sizeof(char), cudaMemcpyHostToDevice));
    clock_t h2dend = clock();

    popcountstart = clock();
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int blockSize = 256;
    int gridSize = (numWindows + blockSize - 1) / blockSize;
    hammingDistanceGlobal << <gridSize, blockSize >> > (d_text, d_pattern, d_dist, L, lengthText);

    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    popcountend = clock();
    printf("GPU Char Comparison (Global) 完成，耗時 %.2f 毫秒\n", elapsed);

    clock_t d2hstart = clock();
    CHECK(cudaMemcpy(h_dist, d_dist, numWindows * sizeof(unsigned char), cudaMemcpyDeviceToHost));
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
            cudaFree(d_pattern);
            cudaFree(d_dist);
            free(h_text);
            free(h_dist);
            return 1;
        }

    cout << "\033[0;37;42m計算正確\033[0m\n";
    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_dist);
    free(h_text);
    free(h_dist);

    return gpuData(readtime, h2d_t, popcount_t, d2h_t, total_t);
}