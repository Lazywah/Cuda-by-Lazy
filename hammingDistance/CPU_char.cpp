#include <iostream>
#include <chrono>
#include <string>

#include <fstream>
#include <sstream>
#include <cstring>

#include "CPU_char.h"
#include "struct.h"

using namespace std;

/*
int hammingDistance(const string& str1, const string& str2) {

    int distance = 0;
    for (size_t i = 0; i < str1.length(); ++i) {
        distance += (str1[i] != str2[i]) ? 1 : 0;
    }
    return distance;
}
*/

/*
void SlidingWindow(const string& T, int* result, int l) {
    int Tsize = T.length();
    int distance = 0;
    string s;
    string p = T.substr(0, l);
    for (int i = 0; i < Tsize - l + 1; i++) {
        s = T.substr(i, l);
        distance = 0;
        for (int j = 0; j < l; j++) {
            distance += (p[j] != s[j]) ? 1 : 0;
        }
        result[i] = distance;
    }
}
*/

void SlidingWindow(const char* T, int* result, int l)
{
    int Tsize = strlen(T);
    for (int i = 0; i < Tsize - l + 1; i++)
    {
        int distance = 0;
        for (int j = 0; j < l; j++)
            distance += (T[j] != T[i + j]) ? 1 : 0;
        result[i] = distance;
    }
}

cpuData CPU_char(const char* filename, int L, int*& answer) {
    clock_t start = clock();
    //const char* filename1 = "Dro64M.txt";
    FILE* infile1;
    errno_t err = fopen_s(&infile1, filename, "r");

    int* result;

    if (err != 0) {
        fprintf(stderr, "\033[0;37;41m無法打開檔案 %s\033[0m\n", filename);
        return cpuData();
    }

    fseek(infile1, 0, SEEK_END);
    long filesize = ftell(infile1);
    fseek(infile1, 0, SEEK_SET);

    char* T = (char*)malloc(filesize + 1);
    if (T == NULL) {
        fprintf(stderr, "\033[0;37;41m記憶體分配失敗\033[0m\n");
        fclose(infile1);
        return cpuData();
    }

    fread(T, 1, filesize, infile1);
    T[filesize] = '\0';

    fclose(infile1);

    clock_t end = clock();

    int Tsize = strlen(T);

    double at = 0;
    int distance = 0;
    int ans_sizes = Tsize - L + 1;
    result = (int*)malloc(ans_sizes * sizeof(int));

    double readtime = (static_cast<double>(end - start) / CLOCKS_PER_SEC);

    clock_t distance_start, distance_end;

    for (int j = 0; j < 1; j++) {
        distance_start = clock();

        SlidingWindow(T, result, L);

        distance_end = clock();
    }

    double distance_time = (static_cast<double>(distance_end - distance_start) / CLOCKS_PER_SEC);
    printf("CPU Char Comparison 完成，耗時 %.2f 毫秒\n", distance_time);

    answer = result;

    printf("\n");
    printf("inputFile: %s\n", filename);
    printf("readtime: %.6f\n", readtime);
    printf("distance: %.6f\n", distance_time);
    printf("total: %.6f\n", readtime + distance_time);
    printf("\n");

    printf("\033[0;37;43m參考答案\033[0m\n");
    printf("dist[] = {");
    for (int i = 0; i < 30; i++)
        printf(" %d%s", result[i], (i==29? " " : ","));
	printf("...}\n");

	return cpuData(readtime, distance_time, readtime + distance_time);
}