#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <windows.h>
#include <limits>

#include "export.h"
/*{
#include <string>
#include <vector>
}*/

using namespace std;


string GetTimestamp()
{
	auto now = chrono::system_clock::now();
	auto in_time_t = chrono::system_clock::to_time_t(now);

	stringstream ss;
	// Format: YYYY-MM-DD_HH-MM-SS
	ss << put_time(localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
	return ss.str();
}


void writeData(ofstream* File, char symbol, const vector<string>& fileSize, int tesTimes, const cpuData*& data, double*& average, int flag)
{
	string title[2] = { "CPU_char", "CPU_char_thread" };
	string type[3] = { "readTime", "distanceTime", "totalTime" };

	if (File != nullptr && File->is_open())
	{
		for (int v = 0; v < 2; v++)
		{
			*File << title[v] << '(' << type[flag] << ')';
			for (int tt = 0; tt < tesTimes; tt++)
				*File << symbol << "test." << tt + 1;
			*File << symbol << "Average" << symbol << "Variance" << symbol << "Standard Deviation";
			*File << "\n";
			for (int f = 0; f < fileSize.size(); f++)
			{
				*File << fileSize[f];
				double timeSum = 0;
				for (int tt = 0; tt < tesTimes; tt++)
				{
					*File << symbol << fixed << setprecision(6) << data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag);
					timeSum += data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag);
					//cout << "v.i.t = " << v * fileSize.size() * tesTimes + f * tesTimes + tt << ", time = " << data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag) << "\n";
				}

				double timeAve = timeSum / tesTimes;
				*File << symbol << fixed << setprecision(6) << timeAve;
				average[v * fileSize.size() + f] = timeAve;

				double timeVar = 0;
				for (int tt = 0; tt < tesTimes; tt++)
					timeVar += pow(data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag) - average[v * fileSize.size() + f], 2);
				timeVar /= tesTimes;
				*File << symbol << fixed << setprecision(6) << timeVar;

				double timeSD = sqrt(timeVar);
				*File << symbol << fixed << setprecision(6) << timeSD;

				*File << "\n";
			}
			*File << "\n\n";
		}
	}
}


void writeData(ofstream* File, char symbol, const vector<string>& fileSize, int tesTimes, const gpuData*& data, double*& average, int flag)
{
	string title[3] = { "GPU_bits", "GPU_char_global", "GPU_char_shared" };
	string type[5] = { "readTime", "h2dTime", "popcounTime", "d2hTime", "totalTime" };

	if (File != nullptr && File->is_open())
	{
		for (int v = 0; v < 3; v++)
		{
			*File << title[v] << '(' << type[flag] << ')';
			for (int tt = 0; tt < tesTimes; tt++)
				*File << symbol << "test." << tt + 1;
			*File << symbol << "Average" << symbol << "Variance" << symbol << "Standard Deviation";
			*File << "\n";
			for (int f = 0; f < fileSize.size(); f++)
			{
				*File << fileSize[f];
				double timeSum = 0;
				for (int tt = 0; tt < tesTimes; tt++)
				{
					*File << symbol << fixed << setprecision(6) << data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag);
					timeSum += data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag);
					//cout << "v.i.t = " << v * fileSize.size() * tesTimes + f * tesTimes + tt << ", time = " << data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag) << "\n";
				}

				double timeAve = timeSum / tesTimes;
				*File << symbol << fixed << setprecision(6) << timeAve;
				average[v * fileSize.size() + f] = timeAve;

				double timeVar = 0;
				for (int tt = 0; tt < tesTimes; tt++)
					timeVar += pow(data[v * fileSize.size() * tesTimes + f * tesTimes + tt].get_time(flag) - average[v * fileSize.size() + f], 2);
				timeVar /= tesTimes;
				*File << symbol << fixed << setprecision(6) << timeVar;

				double timeSD = sqrt(timeVar);
				*File << symbol << fixed << setprecision(6) << timeSD;

				*File << "\n";
			}
			*File << "\n\n";
		}
	}
}

void writeDataAverage(ofstream* File, char symbol, string T, const vector<string>& fileSize, double*& CPU_A, double*& GPU_A)
{
	string title[5] = { "CPU_char", "CPU_char_thread", "GPU_bits", "GPU_char_global", "GPU_char_shared" };

	if (File != nullptr && File->is_open())
	{
		*File << T << "Average(ms)";
		for(int i=0; i<5; i++)
			*File << symbol << title[i];
		*File << "\n";
		for (int f = 0; f < fileSize.size(); f++)
		{
			*File << fileSize[f];
			for (int i = 0; i < 2; i++)
				*File << symbol << fixed << setprecision(6) << CPU_A[i * fileSize.size() + f];
			for (int i = 0; i < 3; i++)
				*File << symbol << fixed << setprecision(6) << GPU_A[i * fileSize.size() + f];
			*File << "\n";
		}
	}
}


void ExportDataWindows(const vector<string>& fileSize, int tesTimes, const cpuData*& CPU_T, const gpuData*& GPU_T, string& baseDirectory)
{
	string folderName = "Data_" + GetTimestamp();
	string fullPath = baseDirectory + "\\" + folderName;

	if (CreateDirectoryA(fullPath.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS)
	{

		cout << "Successfully created new directory: " << fullPath << endl;

		string csvPath = fullPath + "\\Summary_Data.csv";
		string txtPath = fullPath + "\\Summary_Data.txt";

		double* CPU_average = new double[2 * fileSize.size()];
		double* GPU_average = new double[3 * fileSize.size()];
		double* CPU_total_average = new double[2 * fileSize.size()];
		double* GPU_total_average = new double[3 * fileSize.size()];

		ofstream csvFile(csvPath);
		if (csvFile.is_open())
		{
			// CPU export area
			writeData(&csvFile, ',', fileSize, tesTimes, CPU_T, CPU_average, 1);	// distance time
			writeData(&csvFile, ',', fileSize, tesTimes, CPU_T, CPU_total_average, 2);	// total time

			// GPU export area
			writeData(&csvFile, ',', fileSize, tesTimes, GPU_T, GPU_average, 1);	// h2d time
			writeData(&csvFile, ',', fileSize, tesTimes, GPU_T, GPU_average, 2);	// popcount time
			writeData(&csvFile, ',', fileSize, tesTimes, GPU_T, GPU_average, 3);	// d2h time
			writeData(&csvFile, ',', fileSize, tesTimes, GPU_T, GPU_total_average, 4);	// total time

			writeDataAverage(&csvFile, ',', "calculate", fileSize, CPU_average, GPU_average);	// average time
			writeDataAverage(&csvFile, ',', "total", fileSize, CPU_total_average, GPU_total_average);	// average time

			csvFile.close();
			cout << "CSV file exported to directory.\n";
		}
		else
			cerr << "\033[0;30;41mError: Could not create CSV file in directory.\033[0m\n";

		ofstream txtFile(txtPath);
		if (txtFile.is_open())
		{

			// CPU export area
			writeData(&txtFile, '|', fileSize, tesTimes, CPU_T, CPU_average, 1);	// distance time
			writeData(&txtFile, '|', fileSize, tesTimes, CPU_T, CPU_total_average, 2);	// total time

			// GPU export area
			writeData(&txtFile, '|', fileSize, tesTimes, GPU_T, GPU_average, 1);	// h2d time
			writeData(&txtFile, '|', fileSize, tesTimes, GPU_T, GPU_average, 2);	// popcount time
			writeData(&txtFile, '|', fileSize, tesTimes, GPU_T, GPU_average, 3);	// d2h time
			writeData(&txtFile, '|', fileSize, tesTimes, GPU_T, GPU_total_average, 4);	// total time

			writeDataAverage(&txtFile, '|', "calculate", fileSize, CPU_average, GPU_average);	// average time
			writeDataAverage(&txtFile, '|', "total", fileSize, CPU_total_average, GPU_total_average);	// average time

			txtFile.close();
			cout << "TXT file exported to directory.\n";
		}
		else
			cerr << "\033[0;30;41mError: Could not create TXT file in directory.\033[0m\n";

		delete[] CPU_average;
		delete[] GPU_average;

		cout << "\033[0;30;42mFile exporte complete\033[0m\n\n";
	}
	else
		cerr << "\033[0;30;41mError: Could not create directory " << fullPath << ". Error message: " << GetLastError() << "\033[0m\n";
}


bool exportData(const vector<string>& fileSize, int tesTimes, const cpuData* CPU_T, const gpuData* GPU_T)
{
	string YorN;
	cout << "\033[0;30;43mDo you want to export the data? [Y/N]:\033[0m ";
	getline(cin, YorN);
	cout << "\n";
	if (YorN[0] == 'Y' || YorN[0] == 'y')
	{
		string exportBaseDir = "..\\data_HD";
		if (CreateDirectoryA(exportBaseDir.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS)
			ExportDataWindows(fileSize, tesTimes, CPU_T, GPU_T, exportBaseDir);
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
		return 0;
	}
	return 1;
}