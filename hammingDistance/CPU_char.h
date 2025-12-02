#include <iostream>
#include <chrono>
#include <string>

#include <fstream>
#include <sstream>
#include <cstring>

#include "struct.h"

using namespace std;

//int hammingDistance(const string& str1, const string& str2);
//void SlidingWindow(const string& T, int* result, int l);
void SlidingWindow(const char* T, int* result, int l);
cpuData CPU_char(const char* filename1, int L, int*& answer);