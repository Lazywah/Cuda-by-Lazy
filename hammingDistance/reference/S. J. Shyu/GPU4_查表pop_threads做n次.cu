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
    const char* text, UC* d_alphabet, /*ULL* bitText,*/ UC* dist, int L, int lengthText, ULL ref, ULL clearWitnessBit, ULL keepWitnessBit, int threadtimes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bitPerSymbol = 3;
    int numWindows = lengthText - L + 1;
    int i, j, k;
    int pos = (tid)*threadtimes;  // *4
    /*UC val;*/

    if (tid* threadtimes >= numWindows)/* (pos + L + threadtimes >= lengthText || pos + threadtimes >= numWindows) */return;

    // encode text substring
    ULL code = 0;
    char ch;
    
    for (j = 0; j < L-1; j++) {
        /*ch = text[pos + j];*/
        /*ch = text[tid + j];*/
        code = (code << bitPerSymbol) | d_alphabet[text[pos + j]];
    }

    //ULL clearWitnessBit, keepWitnessBit;
    //ULL filterUselessBit = ~0x0ULL;
    //keepWitnessBit = clearWitnessBit = 0x1ULL << (bitPerSymbol - 1);

    //// extend keepWitnessBit for L symbols
    //for (int k = 1; k < L; k++) {
    //    keepWitnessBit = (keepWitnessBit << bitPerSymbol) | clearWitnessBit;
    //}
    //clearWitnessBit = (~keepWitnessBit) & filterUselessBit;

    /*dist[pos] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);*/
    for (i = 0, j = L-1; i < threadtimes; i++, j++)
    {
        code = (code << bitPerSymbol) | d_alphabet[text[pos + j]];

        dist[pos + i] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);
    }
    /*dist[pos] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);*/

    /*ch = text[pos + L];
    if (ch == 'A') val = 0;
    else if (ch == 'C') val = 1;
    else if (ch == 'G') val = 2;
    else if (ch == 'T') val = 3;
    code = (code << bitPerSymbol) | val;
    dist[pos + 1] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);*/
    /*
    ch = text[pos +L+1];
    if (ch == 'A') val = 0;
    else if (ch == 'C') val = 1;
    else if (ch == 'G') val = 2;
    else if (ch == 'T') val = 3;
    code = (code << bitPerSymbol) | val;
    dist[pos + 2] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);
    ch = text[pos + L + 2];
    if (ch == 'A') val = 0;
    else if (ch == 'C') val = 1;
    else if (ch == 'G') val = 2;
    else if (ch == 'T') val = 3;
    code = (code << bitPerSymbol) | val;
    dist[pos + 3] = __popcll(((ref ^ code) + clearWitnessBit) & keepWitnessBit);
    */
}

int main(int argc, char** argv)
{
    string inputFile;
    int L = 6;
    int K = 10;

    if (argc >= 2)
        inputFile = argv[1];
    else
        inputFile = "Dro1024M.txt";

    if (argc >= 4) {
        L = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    // 讀入文字
    clock_t readstart = clock();
    FILE* f = fopen(inputFile.c_str(), "rb");
    if (!f) {
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

    // 編碼程式碼
    UC alphabet[255] = { 0 };
    alphabet['A'] = 0;
    alphabet['C'] = 1;
    alphabet['G'] = 2;
    alphabet['T'] = 3;

    ULL ref = 0;
    int bitPerSymbol = 3;
    char ch;
    /*ULL val = 0;*/
    int i = 0, j = 0, k = 0;
    for (j = 0; j < L; j++) {
        ch = h_text[j];
        ref = (ref << bitPerSymbol) | alphabet[ch];
    }

    //witness
    ULL clearWitnessBit, keepWitnessBit;
    ULL filterUselessBit = ~0x0ULL;
    keepWitnessBit = clearWitnessBit = 0x1ULL << (bitPerSymbol - 1);

    // extend keepWitnessBit for L symbols
    for (k = 1; k < L; k++) {
        keepWitnessBit = (keepWitnessBit << bitPerSymbol) | clearWitnessBit;
    }
    clearWitnessBit = (~keepWitnessBit) & filterUselessBit;

    // 分配 host 記憶體
    UC* h_dist = (UC*)malloc((numWindows+100) * sizeof(UC));
    /*ULL* h_bitText = (ULL*)malloc(numWindows * sizeof(ULL));*/

    // 分配 GPU 記憶體
    char* d_text;
    /*ULL* d_bitText;*/
    UC* d_dist;
    UC* d_alphabet;
    CHECK(cudaMalloc(&d_text, (lengthText+100) * sizeof(char)));

    //CHECK(cudaMalloc(&d_bitText, numWindows * sizeof(ULL)));
    CHECK(cudaMalloc(&d_dist,(numWindows+100) * sizeof(UC)));
    CHECK(cudaMalloc(&d_alphabet, 255 * sizeof(UC)));
    clock_t popcountstart, popcountend;
    clock_t h2dstart = clock();
    CHECK(cudaMemcpy(d_text, h_text, lengthText * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_alphabet, alphabet, 255 * sizeof(UC), cudaMemcpyHostToDevice));
    clock_t h2dend = clock();

    popcountstart = clock();
    // popcount 程式碼
    // 計時開始
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //// Kernel 執行
    //int threadtimes = 8;
    //int blockSize = 1024/ threadtimes;
    //int gridSize = (numWindows + blockSize - 1) / blockSize;
    /*int threadtimes = 8;
    int totalThreads = (numWindows + threadtimes - 1) / threadtimes;
    int blockSize = 128;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;*///能做
    int maxThreadsPerBlock = 1024;
    int threadtimes = 16;  // 假設

    int blockSize = maxThreadsPerBlock / threadtimes;
    /*if (blockSize < 32) blockSize = 32;*/
    int totalThreads = (numWindows + threadtimes - 1) / threadtimes;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    // 
    // kernel 啟動時傳入
    encodeAndHamming << <gridSize, blockSize >> > (d_text, d_alphabet, /*d_bitText,*/ d_dist, L, lengthText, ref, clearWitnessBit, keepWitnessBit, threadtimes);

    // int blockSize = 256;
    // int gridSize = (numWindows + blockSize - 1) / blockSize;
    // encodeAndHamming << <gridSize, blockSize >> > (d_text, d_bitText, d_dist, L, lengthText, ref, clearWitnessBit, keepWitnessBit);


    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("GPU Hamming 計算完成，耗時 %.2f 毫秒\n", elapsed);
    popcountend = clock();



    // 取回結果
    clock_t d2htart = clock();
    CHECK(cudaMemcpy(h_dist, d_dist, numWindows * sizeof(UC), cudaMemcpyDeviceToHost));
    clock_t d2hend = clock();
    double readtime = (double)(readend - readstart) / CLOCKS_PER_SEC;
    double h2d_t = (double)(h2dend - h2dstart) / CLOCKS_PER_SEC;
    double popcount_t = (double)(popcountend - popcountstart) / CLOCKS_PER_SEC;
    double d2h_t = (double)(d2hend - d2htart) / CLOCKS_PER_SEC;
    double total_t = (double)(d2hend - h2dstart) / CLOCKS_PER_SEC;
    /*printf("成功讀入檔案: %s\n", inputFile.c_str());
    printf("讀檔耗時：%.6f 秒\n", readtime);
    printf("2device耗時：%.6f 秒\n", h2d_t);
    printf("Popcount 耗時：%.6f 秒\n", popcount_t);
    printf("2host：%.6f 秒\n", d2h_t);
    printf("total：%.6f 秒\n", total_t);*/
    printf(" %s\n", inputFile.c_str());
    printf("%.6f \n", readtime);
    printf("%.6f \n", h2d_t);
    printf("%.6f \n", popcount_t);
    printf("%.6f \n", d2h_t);
    printf("%.6f \n", total_t);
    // 印前幾個結果
    for (i = 0; i < 30 && i < numWindows; i++) {
        printf("dist[%d] = %d\n", i, h_dist[i]);
    }

    // 清理
    cudaFree(d_text);
    /*cudaFree(d_bitText);*/
    cudaFree(d_dist);
    free(h_text);
    free(h_dist);
    return 0;
}
