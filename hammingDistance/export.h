#include <string>
#include <vector>

#include "struct.h"

using namespace std;

string GetTimestamp(); 
void writeData(ofstream* File, char symbol, const vector<string>& fileSize, int tesTimes, const cpuData*& data, double*& average, int flag);
void writeData(ofstream* File, char symbol, const vector<string>& fileSize, int tesTimes, const gpuData*& data, double*& average, int flag);
void writeDataAverage(ofstream* File, char symbol, const vector<string>& fileSize, double*& CPU_A, double*& GPU_A);
void ExportDataWindows(const vector<string>& fileSize, int tesTimes, const cpuData*& CPU_T, const gpuData*& GPU_T, string& baseDirectory);
bool exportData(const vector<string>& fileSize, int tesTimes, const cpuData* CPU_T, const gpuData* GPU_T);