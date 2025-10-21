#include <stdlib.h>
#include <stdio.h>
#include <numeric>
#include <process.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>  
#include <filesystem>
#include <windows.h>
#include <limits>

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

//////////////////////////////////////////////////////////////////////////

struct Export
{
	int out;
	float time;
	Export(int o=0, float t=0) : out(o), time(t) {};
};

//////////////////////////////////////////////////////////////////////////

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;
Export* list[3];

//////////////////////////////////////////////////////////////////////////

// cudaVersion
__global__ void prescan_arbitrary(int* g_odata, int* g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int* g_odata, int* g_idata, int n, int powerOfTwo);

__global__ void prescan_large(int* g_odata, int* g_idata, int n, int* sums);
__global__ void prescan_large_unoptimized(int* output, int* input, int n, int* sums);

__global__ void add(int* output, int length, int* n1);
__global__ void add(int* output, int length, int* n1, int* n2);

// scan
float sequential_scan(int* output, int* input, int length);
float blockscan(int* output, int* input, int length, bool bcao);
float scan(int* output, int* input, int length, bool bcao);
void scanLargeDeviceArray(int* output, int* input, int length, bool bcao);
void scanSmallDeviceArray(int* d_out, int* d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int* output, int* input, int length, bool bcao);

// printResult
void _checkCudaError(const char* message, cudaError_t err, const char* caller);
void printResult(const char* prefix, int result, long nanoseconds);
void printResult(const char* prefix, int result, float milliseconds);
bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);
long get_nanos();
std::string GetTimestamp();
void ExportDataWindows(int n, int ele[], Export** list, std::string baseDirectory);

//////////////////////////////////////////////////////////////////////////

__global__ void prescan_arbitrary(int* output, int* input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;
	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
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

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
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

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_arbitrary_unoptimized(int* output, int* input, int n, int powerOfTwo) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
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

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
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

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}


__global__ void prescan_large(int* output, int* input, int n, int* sums) {
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
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
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

	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
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

__global__ void prescan_large_unoptimized(int* output, int* input, int n, int* sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
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

	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
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
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}


__global__ void add(int* output, int length, int* n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int* output, int length, int* n1, int* n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

/////////////////////////////////////

float sequential_scan(int* output, int* input, int length) {
	auto start_time = std::chrono::high_resolution_clock::now();

	output[0] = 0; // since this is a prescan, not a scan
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}

	auto end_time = std::chrono::high_resolution_clock::now();;
	auto durationRun = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
	return durationRun;
}

float blockscan(int* output, int* input, int length, bool bcao) {
	int* d_out, * d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void**)&d_out, arraySize);
	cudaMalloc((void**)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int powerOfTwo = nextPowerOfTwo(length);
	if (bcao) {
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
	}

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

float scan(int* output, int* input, int length, bool bcao) {
	int* d_out, * d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void**)&d_out, arraySize);
	cudaMalloc((void**)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	cudaEvent_t start, stop;
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
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

void scanLargeDeviceArray(int* d_out, int* d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
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

void scanSmallDeviceArray(int* d_out, int* d_in, int length, bool bcao) {
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao) {
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
	}
}

void scanLargeEvenDeviceArray(int* d_out, int* d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int* d_sums, * d_incr;
	cudaMalloc((void**)&d_sums, blocks * sizeof(int));
	cudaMalloc((void**)&d_incr, blocks * sizeof(int));

	if (bcao) {
		prescan_large << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

/////////////////////////////////////

void _checkCudaError(const char* message, cudaError_t err, const char* caller) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

void printResult(const char* prefix, int result, long nanoseconds) {
	printf("  ");
	printf(prefix);
	printf(" : %i in %ld ms \n", result, nanoseconds / 1000);
}

void printResult(const char* prefix, int result, float milliseconds) {
	printf("  ");
	printf(prefix);
	printf(" : %i in %f ms \n", result, milliseconds);
}

// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

// from https://stackoverflow.com/a/36095407
long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

std::string GetTimestamp() {
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	// Format: YYYY-MM-DD_HH-MM-SS
	ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
	return ss.str();
}

void ExportDataWindows(int n, int ele[], Export** list, std::string baseDirectory) {
	std::string folderName = "Data_" + GetTimestamp();
	std::string fullPath = baseDirectory + "\\" + folderName;

	if (CreateDirectoryA(fullPath.c_str(), NULL) ||
		GetLastError() == ERROR_ALREADY_EXISTS) {

		std::cout << "Successfully created new directory: " << fullPath << std::endl;

		std::string title[] = { "elements", "CPU Scan(ms)", "GPU Scan(ms)", "Speedup" };

		std::string csvPath = fullPath + "\\Summary_Data.csv";
		std::string txtPath = fullPath + "\\Summary_Data.txt";

		std::ofstream csvFile(csvPath);
		if (csvFile.is_open()) {

			csvFile << title[0] + "," + title[1] + "," + title[2] + "," + title[3] + "\n";
			for (int i = n - 1; i >= 0; i--)
			{
				float speedup = list[0][i].time / list[1][i].time;
				csvFile << std::fixed << std::setprecision(6) << ele[i] << "," << list[0][i].time << "," << list[1][i].time << ",";
				csvFile << std::fixed << std::setprecision(2) << speedup << "\n";
			}

			csvFile.close();
			std::cout << "CSV file exported to directory.\n";
		}
		else {
			std::cerr << "\033[0;30;41mError: Could not create CSV file in directory.\033[0m\n";
		}

		std::ofstream txtFile(txtPath);
		if (txtFile.is_open()) {

			txtFile << title[0] + "|" + title[1] + "|" + title[2] + "|" + title[3] + "\n";
			for (int i = n-1; i >= 0; i--)
			{
				float speedup = list[0][i].time / list[1][i].time;
				txtFile << std::fixed << std::setprecision(6) << ele[i] << "|" << list[0][i].time << "|" << list[1][i].time << "|";
				txtFile << std::fixed << std::setprecision(2) << speedup << "\n";
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

//////////////////////////////////////////////////////////////////////////

void test(int ver, int N) {
	bool canBeBlockscanned = N <= 1024;

	time_t t;
	srand((unsigned)time(&t));
	int* in = new int[N];
	for (int i = 0; i < N; i++) {
		in[i] = rand() % 10;
	}

	printf("%i Elements \n", N);

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

	Export DHost = Export(outHost[N - 1], time_host);
	Export DGPU = Export(outGPU[N - 1], time_gpu);
	Export DGPUBcao = Export(outGPU_bcao[N - 1], time_gpu_bcao);
	
	list[0][ver] = DHost;
	list[1][ver] = DGPU;
	list[2][ver] = DGPUBcao;

	if (canBeBlockscanned) {
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

int main()
{
	/*
	int TEN_MILLION = 10000000;
	int ONE_MILLION = 1000000;
	int TEN_THOUSAND = 10000;

	int elements[] = {
		TEN_MILLION * 2,
		TEN_MILLION,
		ONE_MILLION,
		TEN_THOUSAND,
		5000,
		4096,
		2048,
		2000,
		1000,
		500,
		100,
		64,
		8,
		5
	};
	*/

	int elements[] = {
		16777216, // 1024 * 16384
		8388688, // 1024 * 8192
		4194304, // 1024 * 4096
		2097152, // 1024 * 2048
		1048576, // 1024 * 1024
		524288, // 1024 * 512
		262144, // 1024 * 256
		131072, // 1024 * 128
		65536, // 1024 * 64
		32768, // 1024 * 32
		1024 // 2 ** 10
	};

	int numElements = sizeof(elements) / sizeof(elements[0]);
	for (int i=0; i<3; i++)
		list[i] = new Export[numElements];


	for (int i = 0; i < numElements; i++) {
		test(i, elements[i]);
	}

YorN:
	std::string YorN;
	std::cout << "\033[0;30;43mDo you want to export the data? [Y/N]:\033[0m ";
	//std::cin.ignore('\n');
	getline(std::cin, YorN);
	std::cout << "\n";
	if (YorN[0] == 'Y' || YorN[0] == 'y')
	{
		std::string exportBaseDir = "..\\data_resuction";
		if (CreateDirectoryA(exportBaseDir.c_str(), NULL) ||
			GetLastError() == ERROR_ALREADY_EXISTS) {
				ExportDataWindows(numElements, elements, list, exportBaseDir);
		}
		else {
			std::cerr << "\033[0;30;41mError: Could not create directory " << exportBaseDir << ". Error message: " << GetLastError() << "\033[0m\n";
		}
	}
	else if (YorN[0] == 'N' || YorN[0] == 'n')
	{
	}
	else
	{
		std::cout << "\033[0;30;41mUnrecognized... Please enter again!\033[0m\n\n";
		goto YorN;
	}

	return 0;
}