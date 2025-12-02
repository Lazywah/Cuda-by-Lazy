#include <iostream>
#include <chrono>
#include <cstring>
#include <thread>

#include <fstream>
#include <sstream>

#include "CPU_char_thread.h"
#include "struct.h"

using namespace std;

/*
void SlidingWindowChop(const string& T, int* result, int l, int interval) {
    int Tsize = interval;
    int distance = 0;
    string s;
    string p = T.substr(0, l);
    for (int i = 0; i < Tsize - l + 1; i++) {
        s = T.substr(i, l);
        distance = 0;
        for (int j = 0; j < l; j++)
            distance += (p[j] != s[j]) ? 1 : 0;
        result[i] = distance;
    }
}
*/

void SlidingWindowChop(const char*p, const char* T, int* result, int l, int interval)
{
	int Tsize = interval;
    const char* s;
    int distance;
    //for (int i = 0; i < Tsize - L + 1; i++)
    for (int i = 0; i < Tsize; i++)
    {
        s = T + i;
        distance = 0;
        for (int j = 0; j < l; j++)
            distance += (*(p + j) != *(s + j)) ? 1 : 0;
        result[i] = distance;
    }
}

cpuData CPU_char_thread(const char* filename, int L, int n, int* answer) {
    clock_t start = clock();
    //const char* filename1 = "Dro64M.txt";
    FILE* infile1 = fopen(filename, "r");
    if (infile1 == nullptr) {
        perror("\033[0;37;41mError opening file\033[0m");
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
    int* result = (int*)malloc(ans_sizes * sizeof(int));

    double readtime = (static_cast<double>(end - start) / CLOCKS_PER_SEC);

    clock_t distance_start, distance_end;
    distance_start = clock();

    int rem = ans_sizes % n;
    int block_size = ans_sizes / n;
    int offset = 0;

    thread* t = new thread[n];
    for (int i = 0; i < n; i++)
    {
        //t[i] = thread(SlidingWindowChop, T, T + i * Tsize / n, result + i * (Tsize / n - L + 1), L, Tsize / n);
        int current_block_length = block_size + (i < rem ? 1 : 0);
        t[i] = thread(SlidingWindowChop, T, T + offset, result + offset, L, current_block_length);
        offset += current_block_length;
    }
    for (int i = 0; i < n; i++)
        t[i].join();

    distance_end = clock();

    double distance_time = (static_cast<double>(distance_end - distance_start) / CLOCKS_PER_SEC);
    printf("CPU Thread Char Comparison 完成，耗時 %.2f 毫秒\n", distance_time);

    printf("\n");
    printf("inputFile: %s\n", filename);
    printf("n_core: %d\n", n);
    printf("readtime: %.6f\n", readtime);
    printf("distance: %.6f\n", distance_time);
    printf("total: %.6f\n", readtime + distance_time);
    printf("\n");

    printf("dist[] = {");
    for (int i = 0; i < 30; i++)
        printf(" %d%s", result[i], (i == 29 ? " " : ","));
    printf("...}\n");

    printf("ans[]  = {");
    for (int i = 0; i < 30; i++)
        printf(" %d%s", answer[i], (i == 29 ? " " : ","));
    printf("...}\n");

    for (int i = 0; i < ans_sizes; i++)
        if (result[i] != answer[i])
        {
            cout << "\033[0;37;41m計算錯誤!\033[0m\n";
            cout << "字串長度 Tsize: " << Tsize << "\n";
			cout << "dist 長度: " << ans_sizes << "\n";
            cout << "thread 負責計算個數: " << (Tsize / n) << "\n";
			cout << "在 dist[" << i << "] 發現錯誤: r: " << result[i] << " != a: " << answer[i] << "\n";
            free(result);
            return 1;
        }

    cout << "\033[0;37;42m計算正確\033[0m\n";
    free(result);

	return cpuData(readtime, distance_time, readtime + distance_time);
}