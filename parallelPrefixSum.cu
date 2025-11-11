///////////////////////////////////////////////////////////////////////////

// 測試說明區
/*

測試目標：
測試使用'CPU'和使用'GPU'(優化、未優化、內建)對於並列前綴和的運行速度並比對

結果目標：
做出多組數據之運算速度變化，CPU及GPU之運算速度差距及分水嶺(何組數據以下CPU較快、GPU較快)

測試細節：
---測試流程---
測試主要分組：
1. 串行 CPU 掃描
2. 多 Block Blelloch 掃描	(未優化)
3. 多 Block Blelloch 掃描	(BCAO 優化)
4. Thrust 函式庫 (Exclusive Scan)
5. CUB 函式庫	(Device Scan)		(未知原因出錯，暫時擱置)

(額外測試)
6. 單 Block Blelloch 掃描	(未優化)		(elements <= 1024)(不紀錄數據)
7. 單 Block Blelloch 掃描	(BCAO 優化)	(elements <= 1024)(不紀錄數據)

數據紀錄格式：
1. elements | CPU test.* | average | Variance | Standard Deviation
2. elements | GPU test.* | average | Variance | Standard Deviation
3. elements | GPU_bcao test.* | average | Variance | Standard Deviation
4. elements | Built-in test.* | average | Variance | Standard Deviation

test()：
1. 創造亂數矩陣
2. 呼叫各版本測試函數
3. 記錄入list中
4. 返回Main決定是否匯出數據

各測試版本流程簡述：

1. CPU:
	* 單一串行階段：在 Host (CPU) 上循序執行 $O(N)$ 複雜度的前綴和計算。用作所有並行版本的性能基線。

2. 多 Block Blelloch 掃描 (未優化 \ BCAO 優化):
	* 三階段 GPU 運算：
	* 1. Block 內掃描：每個 Block 獨立執行 Blelloch 演算法計算結果，並提取 Block 總和 (Si)。
	* 2. Offset 計算：對 Si 陣列再次掃描，計算出最終偏移量。
	* 3. 最終更新：啟動第三個核心將偏移量加回各 Block 結果中。
	*
	* BCAO 優化：
	* 1. 在三階段運算的基礎上使用 Bank Conflict Avoidance Offset (BCAO) 機制。
	* 2. 透過調整共享記憶體存取索引來避免記憶體體衝突的優化技術。

3. Thrust 函式庫 (Exclusive Scan):
	* 高階自動化：
	* Thrust 自動處理數據傳輸、記憶體管理，並在內部自動選擇並執行針對 GPU 優化的多階段並行掃描核心。

4. CUB 函式庫 (Device Scan)
	* NVIDIA 最佳化：
	* CUB 是 NVIDIA 提供的原始建構單元，執行經過深度優化、針對硬體特性調整的並行掃描算法，通常在大型數據集上能展現最佳的 GPU 性能。

(額外)
5. 單 Block Blelloch 掃描 (未優化 \ BCAO 優化):
	* 單 Block 階段：
	* 1. 僅使用 一個 Block 啟動 未優化 \ BCAO 優化 的單階段掃描核心。
	* 2. 這種情況下，無需進行第二階段的偏移計算。

*/
//////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <string>

// kernels.cuh
#include <device_launch_parameters.h>
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

// scan.cuh
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>

// export.h
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <windows.h>
#include <limits>

using namespace std;

//////////////////////////////////////////////////////////////////////////

// 全域變數宣告

// scan.cuh
int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

//////////////////////////////////////////////////////////////////////////

// 函數宣告


// main.cpp
void testWithoutPrint(int N);
void test(int e_curr, int numOfElements, int tt_curr, int tesTimes, float* list, int N);


// kernels.cuh
__global__ void prescan_arbitrary(int* g_odata, int* g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int* g_odata, int* g_idata, int n, int powerOfTwo);

__global__ void prescan_large(int* g_odata, int* g_idata, int n, int* sums);
__global__ void prescan_large_unoptimized(int* output, int* input, int n, int* sums);

__global__ void add(int* output, int length, int* n);
__global__ void add(int* output, int length, int* n1, int* n2);


// scan.cuh
float sequential_scan(int* output, int* input, int length);
float blockscan(int* output, int* input, int length, bool bcao);
float scan(int* output, int* input, int length, bool bcao);
void scanLargeDeviceArray(int* output, int* input, int length, bool bcao);
void scanSmallDeviceArray(int* d_out, int* d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int* output, int* input, int length, bool bcao);
float thrust_exclusive_scan(int* host_output, const int* host_input, size_t N);
float cub_device_exclusive_scan(int* d_output, int* d_input, int N);


// utils.h
void _checkCudaError(const char* message, cudaError_t err, const char* caller);
void printResult(const char* prefix, int result, long nanoseconds);
void printResult(const char* prefix, int result, float milliseconds);
bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);
long get_nanos();


// export.h
string GetTimestamp();
void ExportDataWindows(int versionCount, int numOfElements, const int* elements, int tesTime, const float* list, string baseDirectory);
bool exportData(int versionCount, int numOfElements, const int* elements, int tesTime, const float* list);

//////////////////////////////////////////////////////////////////////////



/*///////////////////////////////////*/
/*             main.cpp              */
/*///////////////////////////////////*/

int main()
{
	// CPU | GPU | GPU_bcao | thrust | cub
	int version = 4;

	// 訂定數據量 (2^10 ~ 2^24)
	int elements[] =
	{
		1024, // 2 ^ 10
		2048, // 1024 * 2
		4096, // 1024 * 4
		8192, // 1024 * 8
		16384, // 1024 * 16
		32768, // 1024 * 32
		65536, // 1024 * 64
		131072, // 1024 * 128
		262144, // 1024 * 256
		524288, // 1024 * 512
		1048576, // 1024 * 1024
		2097152, // 1024 * 2048
		4194304, // 1024 * 4096
		8388688, // 1024 * 8192
		16777216 // 1024 * 16384
	};

	int tesTimes = 5;


	// 矩陣動態宣告(用於紀錄數據)
	// 原:[version][elements][tesTimes] -> 進行線性化(方便傳遞)
	// 公式：V*elements*tesTimes + E*tesTimes + T
	int numOfElements = sizeof(elements) / sizeof(elements[0]);
	float* list = new float[version * numOfElements * tesTimes];

	// 由於在運行第一次時，誤差較大，故先運行一次(不記錄數據)
	testWithoutPrint(elements[0]);

	for (int j = 0; j < tesTimes; j++)
	{
		printf("\033[0;30;43m----Test.%d----\033[0m\n", j + 1);
		for (int i = 0; i < numOfElements; i++)
		{
			printf("\033[32m----Elements = %d----\033[0m\n", elements[i]);
			test(i, numOfElements, j, tesTimes, list, elements[i]);
		}
	}


	// 呼叫數據匯出
	while (!exportData(version, numOfElements, elements, tesTimes, list));

	delete[] list;

	return 0;
}

// 由於在運行第一次時，誤差較大，故先運行一次(不記錄數據)
void testWithoutPrint(int N)
{
	time_t t;
	srand((unsigned)time(&t));
	int* in = new int[N];
	for (int i = 0; i < N; i++)
		in[i] = rand() % 10;

	int* outHost = new int[N]();
	float time_host = sequential_scan(outHost, in, N);

	int* outGPU = new int[N]();
	float time_gpu = scan(outGPU, in, N, false);

	int* outGPU_bcao = new int[N]();
	float time_gpu_bcao = scan(outGPU_bcao, in, N, true);

	int* outGPU_thrust = new int[N]();
	float time_gpu_thrust = thrust_exclusive_scan(outGPU_thrust, in, N);

	//int* outGPU_cub = new int[N]();
	//float time_gpu_cub = cub_device_exclusive_scan(outGPU_cub, in, N);

	delete[] in;
	delete[] outHost;
	delete[] outGPU;
	delete[] outGPU_bcao;
}

// 實際呼叫函數測試區
// 公式：V*elements*tesTimes + E*tesTimes + T
// -> (0~4)*ele*tesT + e*tesT + tt
void test(int e, int ele, int tt, int tesT, float* list, int N)
{
	// 確認數據量，用於判定是否執行直接呼叫(跳過分block過程)版本
	bool canBeBlockScanned = (N <= 1024);

	// 隨機數據區
	time_t t;
	srand((unsigned)time(&t));
	int* in = new int[N];
	for (int i = 0; i < N; i++)
		in[i] = rand() % 10;

	// 函數呼叫區
	// sequential scan on CPU
	int* outHost = new int[N]();
	float time_host = sequential_scan(outHost, in, N);
	printResult("host    ", outHost[N - 1], time_host);

	// full scan
	int* outGPU = new int[N]();
	float time_gpu = scan(outGPU, in, N, false);
	printResult("gpu     ", outGPU[N - 1], time_gpu);

	// full scan with BCAO
	int* outGPU_bcao = new int[N]();
	float time_gpu_bcao = scan(outGPU_bcao, in, N, true);
	printResult("gpu bcao", outGPU_bcao[N - 1], time_gpu_bcao);

	// thrust_exclusive_scan
	int* outGPU_thrust = new int[N]();
	float time_gpu_thrust = thrust_exclusive_scan(outGPU_thrust, in, N);
	printResult("gpu thrust", outGPU_thrust[N - 1], time_gpu_thrust);

	// cub_device_exclusive_scan
	//int* outGPU_cub = new int[N]();
	//float time_gpu_cub = cub_device_exclusive_scan(outGPU_cub, in, N);
	//printResult("gpu cub", outGPU_cub[N - 1], time_gpu_cub);

	list[0 * ele * tesT + e * tesT + tt] = time_host;
	list[1 * ele * tesT + e * tesT + tt] = time_gpu;
	list[2 * ele * tesT + e * tesT + tt] = time_gpu_bcao;
	list[3 * ele * tesT + e * tesT + tt] = time_gpu_thrust;
	//list[4 * ele * tesT + e * tesT + tt] = time_gpu_cub;

	if (canBeBlockScanned)
	{
		// basic level 1 block scan
		int* out_1block = new int[N]();
		float time_1block = blockscan(out_1block, in, N, false);
		printResult("level 1 ", out_1block[N - 1], time_1block);

		// level 1 block scan with BCAO
		int* out_1block_bcao = new int[N]();
		float time_1block_bcao = blockscan(out_1block_bcao, in, N, true);
		printResult("l1 bcao ", out_1block_bcao[N - 1], time_1block_bcao);

		delete[] out_1block;
		delete[] out_1block_bcao;
	}

	printf("\n");

	delete[] in;
	delete[] outHost;
	delete[] outGPU;
	delete[] outGPU_bcao;
}



/*///////////////////////////////////*/
/*            kernels.cu             */
/*///////////////////////////////////*/

__global__ void prescan_arbitrary(int* output, int* input, int n, int powerOfTwo)
{
	// allocated on invocation
	extern __shared__ int temp[];

	int threadID = threadIdx.x;
	int ai = threadID;
	int bi = threadID + (n / 2);

	// 為共享記憶體索引添加一個特定的偏移量，以確保連續的執行緒存取
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	// load input into shared memory
	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}

	// build sum in place up the tree
	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (threadID == 0)
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;

	// traverse down tree & build scan
	for (int d = 1; d < powerOfTwo; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// write results to device memory
	if (threadID < n)
	{
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}


__global__ void prescan_arbitrary_unoptimized(int* output, int* input, int n, int powerOfTwo)
{
	// allocated on invocation
	extern __shared__ int temp[];

	int threadID = threadIdx.x;

	// load input into shared memory
	if (threadID < n)
	{
		temp[2 * threadID] = input[2 * threadID];
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else
	{
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}

	// build sum in place up the tree
	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (threadID == 0)
		temp[powerOfTwo - 1] = 0;

	// traverse down tree & build scan
	for (int d = 1; d < powerOfTwo; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// write results to device memory
	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID];
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}


__global__ void prescan_large(int* output, int* input, int n, int* sums)
{
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();

	if (threadID == 0)
	{
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}


__global__ void prescan_large_unoptimized(int* output, int* input, int n, int* sums)
{
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID + 1)];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();

	if (threadID == 0)
	{
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID + 1)] = temp[2 * threadID + 1];
}


__global__ void add(int* output, int length, int* n)
{
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}


__global__ void add(int* output, int length, int* n1, int* n2)
{
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}



/*///////////////////////////////////*/
/*              scan.cu              */
/*///////////////////////////////////*/

float sequential_scan(int* output, int* input, int length)
{
	auto start = chrono::high_resolution_clock::now();

	/*
	 * 專屬前綴和 (Exclusive Prefix Sum / Exclusive Scan) 定義：
	 * 1. 目標：計算陣列中每個位置之前 (不含該位置本身) 所有元素的累積總和。
	 * 2. 數學表示：output[i] = Sum(input[k]) for k from 0 to i-1
	 * 3. 邊界條件：output[0] 預設為 0 (因為 input[0] 之前沒有元素)。
	 * 4. 應用意義：此結果常被用作串流壓縮、分塊任務分配等並行演算法中的「偏移量 (Offset)」。
	 *
	 * -- 解釋 by Gemini (2.5 Flash)
	 */
	output[0] = 0;
	for (int i = 1; i < length; i++)
		output[i] = input[i - 1] + output[i - 1];

	auto end = chrono::high_resolution_clock::now();
	auto durationRun = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
	return durationRun;
}


float blockscan(int* output, int* input, int length, bool bcao)
{
	int* d_out, * d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void**)&d_out, arraySize);
	cudaMalloc((void**)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// 基於 powerOfTwo（通常是略大於 length 的 2 的冪次）
	// 乘以 2 是因為在樹狀掃描演算法中
	// share memory 通常需要 2 倍於數據的空間來儲存節點和避免衝突。
	int powerOfTwo = nextPowerOfTwo(length);
	if (bcao) // 優化
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
	else // 未優化
		prescan_arbitrary_unoptimized << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}


float scan(int* output, int* input, int length, bool bcao)
{
	int* d_out, * d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void**)&d_out, arraySize);
	cudaMalloc((void**)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	cudaEvent_t start, stop;
	float elapsedTime = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		scanSmallDeviceArray(d_out, d_in, length, bcao);
	}

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

/* 以下三段程式碼共同構成了一個完整且分層級的 Device-Wide Prefix Scan 實現。
 *
 * 這套分層設計展示了高性能並行掃描的標準結構：
 * 1. 分治法： 將大陣列分解為多個 Block。
 * 2. 核心優化： 在 Block 內，使用共享記憶體和體衝突優化(bcao) 來實現最快的速度。
 * 3. 二次掃描： 建立中間的「總和陣列」，對它進行第二次掃描來計算全域偏移量。
 * 4. 邊界處理： 專門處理非 2 的冪次(nextPowerOfTwo) 和非整數倍 Block 大小的數據(scanLargeDeviceArray)，確保所有數據都被正確處理。
 *
 * -- 解釋 by Gemini (2.5 Flash)
 */

 // 用於處理大於 ELEMENTS_PER_BLOCK(1024) 的 Array
void scanLargeDeviceArray(int* d_out, int* d_in, int length, bool bcao)
{
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (!remainder)
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int* startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

		add << <1, remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}


// 用於處理未滿 ELEMENTS_PER_BLOCK(1024) 或餘下的 Array
void scanSmallDeviceArray(int* d_out, int* d_in, int length, bool bcao)
{
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao)
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
	else
		prescan_arbitrary_unoptimized << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
}


// 用於處理等於 ELEMENTS_PER_BLOCK(1024) 的 Array
void scanLargeEvenDeviceArray(int* d_out, int* d_in, int length, bool bcao)
{
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int* d_sums, * d_incr;
	cudaMalloc((void**)&d_sums, blocks * sizeof(int));
	cudaMalloc((void**)&d_incr, blocks * sizeof(int));

	if (bcao)
		prescan_large << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	else
		prescan_large_unoptimized << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK)
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	else
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);

	add << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}


float thrust_exclusive_scan(int* host_output, const int* host_input, size_t N)
{
	int* d_input, * d_output;
	size_t arraySize = N * sizeof(int);

	cudaMalloc((void**)&d_input, arraySize);
	cudaMalloc((void**)&d_output, arraySize);

	cudaMemcpy(d_input, host_input, arraySize, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// 使用 thrust::device_ptr 包裝原始指標，以便 Thrust 演算法能識別 GPU 記憶體
	thrust::device_ptr<int> dev_input_ptr(d_input);
	thrust::device_ptr<int> dev_output_ptr(d_output);

	thrust::exclusive_scan(dev_input_ptr, dev_input_ptr + N, dev_output_ptr, 0);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(host_output, d_output, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}


// FIXME
float cub_device_exclusive_scan(int* output, int* input, int N)
{
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	// 2. 第一次呼叫：確定所需的工作空間大小
	// N：元素數量
	// cudaStream_t(0)：使用預設串流
	cub::DeviceScan::template ExclusiveSum(d_temp_storage,
		temp_storage_bytes,
		input,
		output,
		N,
		(cudaStream_t)0,
		0);

	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cudaEvent_t start, stop;
	float elapsedTime = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// 4. 第二次呼叫：執行實際的掃描計算
	cub::DeviceScan::template ExclusiveSum(d_temp_storage,
		temp_storage_bytes,
		input,
		output,
		N,
		(cudaStream_t)0,
		0);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaFree(d_temp_storage);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}



/*///////////////////////////////////*/
/*             utils.cpp             */
/*///////////////////////////////////*/

void _checkCudaError(const char* message, cudaError_t err, const char* caller)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}


void printResult(const char* prefix, int result, long nanoseconds)
{
	printf("  ");
	printf(prefix);
	printf(" : %i in %ld ms \n", result, nanoseconds / 1000);
}


void printResult(const char* prefix, int result, float milliseconds)
{
	printf("  ");
	printf(prefix);
	printf(" : %i in %f ms \n", result, milliseconds);
}


// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x)
{
	return x && !(x & (x - 1));
}


// from https://stackoverflow.com/a/12506181
//int nextPowerOfTwo(int x) 
//{
//	int power = 1;
//	while (power < x)
//		power *= 2;
//	return power;
//}


// 改為使用位元運算(較快速)
int nextPowerOfTwo(int x)
{
	if (x <= 0)
		return 1;
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x++;
	return x;
}


// from https://stackoverflow.com/a/36095407
long get_nanos()
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}



/*///////////////////////////////////*/
/*             export.cpp            */
/*///////////////////////////////////*/

string GetTimestamp()
{
	auto now = chrono::system_clock::now();
	auto in_time_t = chrono::system_clock::to_time_t(now);

	stringstream ss;
	// Format: YYYY-MM-DD_HH-MM-SS
	ss << put_time(localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
	return ss.str();
}


void ExportDataWindows(int ver, int num, const int* ele, int tesT, const float* list, string baseDirectory)
{
	string folderName = "Data_" + GetTimestamp();
	string fullPath = baseDirectory + "\\" + folderName;

	if (CreateDirectoryA(fullPath.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS)
	{

		cout << "Successfully created new directory: " << fullPath << endl;

		string title[] = { "elements", "CPU Scan(ms)", "GPU Scan(ms)", "GPU_bcao Scan(ms)", "thrust Scan(ms) (built-in)", "cub Scan(ms) (built-in)" };

		string csvPath = fullPath + "\\Summary_Data.csv";
		string txtPath = fullPath + "\\Summary_Data.txt";

		ofstream csvFile(csvPath);
		if (csvFile.is_open())
		{

			// Title,test.*,average,Variance,Standard Deviation
			for (int v = 0; v < ver; v++)
			{
				csvFile << title[v + 1];
				for (int i = 1; i <= tesT; i++)
					csvFile << ",test." << i;
				csvFile << ",Average,Variance,Standard Deviation\n";

				for (int e = 0; e < num; e++)
				{
					csvFile << ele[e];

					// test data
					float timeSum = 0.0f;
					for (int tt = 0; tt < tesT; tt++)
					{
						csvFile << "," << fixed << setprecision(6) << list[v * num * tesT + e * tesT + tt];
						timeSum += list[v * num * tesT + e * tesT + tt];
					}

					// data average
					float timeAve = timeSum / tesT;
					csvFile << "," << fixed << setprecision(6) << timeAve;

					// data variance
					float timeVar = 0.0f;
					for (int tt = 0; tt < tesT; tt++)
						timeVar += pow(list[v * num * tesT + e * tesT + tt] - timeAve, 2);
					timeVar /= tesT;
					csvFile << "," << fixed << setprecision(6) << timeVar;

					// data Standard Deviation
					float timeSD = sqrt(timeVar);
					csvFile << "," << fixed << setprecision(6) << timeSD;

					csvFile << "\n";
				}
				csvFile << "\n\n";
			}

			csvFile.close();
			cout << "CSV file exported to directory.\n";
		}
		else
			cerr << "\033[0;30;41mError: Could not create CSV file in directory.\033[0m\n";

		ofstream txtFile(txtPath);
		if (txtFile.is_open())
		{

			// Title | test.* | average | Variance | Standard Deviation
			for (int v = 0; v < ver; v++)
			{
				txtFile << title[v + 1];
				for (int i = 1; i <= tesT; i++)
					txtFile << "|test." << i;
				txtFile << "|Average|Variance|Standard Deviation\n";

				txtFile << "--";
				for (int i = 1; i < tesT + 4; i++)
					txtFile << "|--";

				txtFile << "\n";

				for (int e = 0; e < num; e++)
				{
					txtFile << ele[e];

					// test data
					float timeSum = 0.0f;
					for (int tt = 0; tt < tesT; tt++)
					{
						txtFile << "|" << fixed << setprecision(6) << list[v * num * tesT + e * tesT + tt];
						timeSum += list[v * num * tesT + e * tesT + tt];
					}

					// 平均數(data average)
					float timeAve = timeSum / tesT;
					txtFile << "|" << fixed << setprecision(6) << timeAve;

					// 變異數(data variance)
					float timeVar = 0.0f;
					for (int tt = 0; tt < tesT; tt++)
						timeVar += pow(list[v * num * tesT + e * tesT + tt] - timeAve, 2);
					timeVar /= tesT;
					txtFile << "|" << fixed << setprecision(6) << timeVar;

					// 標準差(data Standard Deviation)
					float timeSD = sqrt(timeVar);
					txtFile << "|" << fixed << setprecision(6) << timeSD;

					txtFile << "\n";
				}
				txtFile << "\n\n";
			}

			txtFile.close();
			cout << "TXT file exported to directory.\n";
		}
		else
			cerr << "\033[0;30;41mError: Could not create TXT file in directory.\033[0m\n";

		cout << "\033[0;30;42mFile exported successfully\033[0m\n\n";
	}
	else
		cerr << "\033[0;30;41mError: Could not create directory " << fullPath << ". Error message: " << GetLastError() << "\033[0m\n";
}


bool exportData(int ver, int num, const int* ele, int tesT, const float* list)
{
	string YorN;
	cout << "\033[0;30;43mDo you want to export the data? [Y/N]:\033[0m ";
	getline(cin, YorN);
	cout << "\n";
	if (YorN[0] == 'Y' || YorN[0] == 'y')
	{
		string exportBaseDir = "..\\data_PPS";
		if (CreateDirectoryA(exportBaseDir.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS)
			ExportDataWindows(ver, num, ele, tesT, list, exportBaseDir);
		else
			cerr << "\033[0;30;41mError: Could not create directory " << exportBaseDir << ". Error message: " << GetLastError() << "\033[0m\n";
	}
	else if (YorN[0] == 'N' || YorN[0] == 'n')
	{
		// do nothing
	}
	else
	{
		cout << "\033[0;30;41mUnrecognized... Please enter again!\033[0m\n\n";
		return 0; // 錯誤
	}
	return 1; // 完成
}