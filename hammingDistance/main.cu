#include <iostream>
#include <vector>
#include <iomanip>

#include "struct.h"
#include "CPU_char.h"
#include "CPU_char_thread.h"
#include "GPU_bits.cuh"
#include "GPU_char_global.cuh"
#include "GPU_char_shared.cuh"
#include "export.h"

int main(int argc, char** argv)
{
	//string inputFile;
	std::vector<string> inputSize = {	"1K",
										"1M",
										"2M",
										"4M",
										"8M",
										"16M",
										"32M",
										"64M",
										"128M"	};
	int L = 6;
	int K = 10;
	int tesTimes = 5;

	//std::vector<std::vector<std::vector<cpuData>>> CPU_times(2,
	//			std::vector<std::vector<cpuData>>  (inputSize.size(),
	//						std::vector<cpuData>   (tesTimes)));

	//std::vector<std::vector<std::vector<gpuData>>> GPU_times(3,
	//			std::vector<std::vector<gpuData>>  (inputSize.size(),
	//						std::vector<gpuData>   (tesTimes)));

	cpuData* CPU_times = new cpuData[2 * inputSize.size() * tesTimes];
	gpuData* GPU_times = new gpuData[3 * inputSize.size() * tesTimes];

	/*
	if (argc >= 2)
		inputFile = argv[1];
	else
		inputFile = "./dataGenerator/Dro1024M.txt";
	*/

	if (argc >= 4)
	{
		L = atoi(argv[2]);
		K = atoi(argv[3]);
	}

	unsigned int n_cores = std::thread::hardware_concurrency();

	string inputFile = "./testData/text/DNA" + inputSize[0] + ".txt";
	bits_hammingDistance_noPrint(inputFile, L, K);

	for (int t = 0; t < tesTimes; t++)
		for (int i = 0; i < inputSize.size(); i++)
		{
			int* answer = nullptr;

			inputFile = "./testData/text/DNA" + inputSize[i] + ".txt";

			CPU_times[0 * inputSize.size() * tesTimes + i * tesTimes + t] = CPU_char(inputFile.c_str(), L, answer);
			std::cout << "\n已存入[v.i.t] = " << "[" << 0 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << &CPU_times[0 * inputSize.size() * tesTimes + i * tesTimes + t].distance_t;
			std::cout << "\n已存入[v.i.t] = " << "[" << 0 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << fixed << setprecision(6) << CPU_times[0 * inputSize.size() * tesTimes + i * tesTimes + t].distance_t << "\n";
			std::cout << "\n\n----------------------------------------\n\n";
			CPU_times[1 * inputSize.size() * tesTimes + i * tesTimes + t] = CPU_char_thread(inputFile.c_str(), L, n_cores, answer);
			std::cout << "\n已存入[v.i.t] = " << "[" << 1 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << &CPU_times[1 * inputSize.size() * tesTimes + i * tesTimes + t].distance_t;
			std::cout << "\n已存入[v.i.t] = " << "[" << 1 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << fixed << setprecision(6) << CPU_times[1 * inputSize.size() * tesTimes + i * tesTimes + t].distance_t << "\n";
			std::cout << "\n\n----------------------------------------\n\n";

			GPU_times[0 * inputSize.size() * tesTimes + i * tesTimes + t] = GPU_bits_hammingDistance(inputFile, L, K, answer);
			std::cout << "\n已存入[v.i.t] = " << "[" << 0 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << &GPU_times[0 * inputSize.size() * tesTimes + i * tesTimes + t].popcount_t;
			std::cout << "\n已存入[v.i.t] = " << "[" << 0 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << fixed << setprecision(6) << GPU_times[0 * inputSize.size() * tesTimes + i * tesTimes + t].popcount_t;
			std::cout << "\n\n----------------------------------------\n\n";
			GPU_times[1 * inputSize.size() * tesTimes + i * tesTimes + t] = GPU_char_hammingDistance_global(inputFile, L, K, answer);
			std::cout << "\n已存入[v.i.t] = " << "[" << 1 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << &GPU_times[1 * inputSize.size() * tesTimes + i * tesTimes + t].popcount_t;
			std::cout << "\n已存入[v.i.t] = " << "[" << 1 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << fixed << setprecision(6) << GPU_times[1 * inputSize.size() * tesTimes + i * tesTimes + t].popcount_t << "\n";
			std::cout << "\n\n----------------------------------------\n\n";
			GPU_times[2 * inputSize.size() * tesTimes + i * tesTimes + t] = GPU_char_hammingDistance_shared(inputFile, L, K, answer);
			std::cout << "\n已存入[v.i.t] = " << "[" << 2 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << &GPU_times[2 * inputSize.size() * tesTimes + i * tesTimes + t].popcount_t;
			std::cout << "\n已存入[v.i.t] = " << "[" << 2 * inputSize.size() * tesTimes + i * tesTimes + t << "], time = " << fixed << setprecision(6) << GPU_times[2 * inputSize.size() * tesTimes + i * tesTimes + t].popcount_t << "\n";
			std::cout << "\n\n----------------------------------------\n\n";

			free(answer);
		}

	exportData(inputSize, tesTimes, CPU_times, GPU_times);
}