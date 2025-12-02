#include <iostream>
#include <chrono>
#include <cstring>
#include <thread>

#include <fstream>
#include <sstream>

#include "struct.h"

using namespace std;

//void SlidingWindowChop(const string& T, int* result, int l, int interval);
void SlidingWindowChop(const char* p, const char* T, int* result, int l, int interval);
cpuData CPU_char_thread(const char* filename1, int L, int n, int* answer);