#include <iostream>
#include <chrono>
#include <string>

#include <fstream>
#include <sstream>
#include <cstring>

using namespace std;

// 計算簡易漢明距離
int hammingDistance(const string& str1, const string& str2) {

    int distance = 0;
    for (size_t i = 0; i < str1.length(); ++i) {
        distance += (str1[i] != str2[i]) ? 1 : 0;
    }
    return distance;
}

void SlidingWindow(const string& T, int* result, int l) {
    int Tsize = T.length();
    int distance = 0;
    string s;
    //for (int k = 0; k < Tsize - l; k++) {//every substring
    string p = T.substr(0, l);//k or 0
    //cout << p << endl;
    for (int i = 0; i < Tsize - l + 1; i++) {//k or 0
        s = T.substr(i, l);
        //cout << i << " :" << s << endl;
        distance = 0;
        for (int j = 0; j < l; j++) {
            distance += (p[j] != s[j]) ? 1 : 0;
        }
        result[i] = distance;
    }
    //}
}

int main() {
    // 開始計時

    for (int l = 6; l < 7; l += 4) {

        auto start1_time = std::chrono::high_resolution_clock::now();
        clock_t start1 = clock();
        // 檔案名稱
        const char* filename1 = "Dro64M.txt";
        FILE* infile1;
        errno_t err = fopen_s(&infile1, filename1, "r");  // 使用 fopen_s

        if (err != 0) {
            fprintf(stderr, "無法打開檔案 %s\n", filename1);
            return 1;
        }

        // 移動到檔案的結尾，獲取檔案大小
        fseek(infile1, 0, SEEK_END);
        long filesize = ftell(infile1);  // 獲取檔案大小
        fseek(infile1, 0, SEEK_SET);     // 返回到檔案的開頭

        // 分配內存來保存檔案內容
        char* T = (char*)malloc(filesize + 1);  // +1 是為了放置字串結尾符 '\0'
        if (T == NULL) {
            fprintf(stderr, "記憶體分配失敗\n");
            fclose(infile1);
            return 1;
        }

        // 讀取檔案內容到 T 陣列中
        fread(T, 1, filesize, infile1);
        T[filesize] = '\0';  // 確保以空字元結尾

        // 關閉檔案
        fclose(infile1);

        auto end1_time = std::chrono::high_resolution_clock::now();
        clock_t end1 = clock();

        int Tsize = strlen(T);  // 計算字串 T 的長度

        double at = 0;
        int distance = 0;
        int ans_sizes = Tsize - l + 1;
        int* result = (int*)malloc(ans_sizes * sizeof(int));


        std::chrono::duration<double, std::milli> readtime = end1_time - start1_time;
        double readtime1 = (static_cast<double>(end1 - start1) / CLOCKS_PER_SEC);


        std::chrono::high_resolution_clock::time_point distance_start, distance_end;
        clock_t distance_start1, distance_end1;
        for (int j = 0; j < 1; j++) {
            // 開始計時
            distance_start = chrono::high_resolution_clock::now();
            distance_start1 = clock();
            // 計算漢明距離
            SlidingWindow(T, result, l);
            // 結束計時
            distance_end = chrono::high_resolution_clock::now();
            distance_end1 = clock();

            // 輸出結果

            //std::ofstream outFile("output.txt");
            //// 檢查文件是否成功打開
            //if (outFile.is_open()) {
            //    // 將陣列的每個元素寫入文件，並用空格分隔
            //    for (int i = 0; i < ans_sizes; ++i) {
            //        outFile << result[i];
            //    }
            //    outFile.close(); // 完成後關閉文件
            //    std::cout << "陣列已成功寫入 output.txt 文件！" << std::endl;
            //}
            //else {
            //    std::cout << "無法打開文件。" << std::endl;
            //}

        }
        //cout << " T : " << T << endl;

        ////print 每個L-substring
        //int count = Tsize - l + 1;  // 初始換行間隔為8
        //int index = 0;  // 用來跟蹤元素位置
        //while (index < result.size()) {
        //    // 打印 `count` 個元素
        //    for (int i = 0; i < count && index < result.size(); ++i) {
        //        std::cout << result[index] << " ";
        //        ++index;
        //    }

        //    std::cout << std::endl;  // 換行

        //    // 下一次的換行間隔遞減
        //    if (count > 1) {
        //        --count;
        //    }
        //}

        //for (int i = 0; i < ans_sizes; ++i) {// one substring
        //    cout << result[i] << " ";
        //}
        std::chrono::duration<double, std::milli> distance_time = distance_end - distance_start;
        double distance_time1 = (static_cast<double>(distance_end1 - distance_start1) / CLOCKS_PER_SEC);
        //printf("l= %2d, %f, %f\n", l, readtime.count(), distance_time.count());
        for (int i = 0; i < 30; i++)
            printf("dist[%d] = %d\n", i, result[i]);
        printf("%.6f\n",readtime1);
        printf("%.6f\n",distance_time1);
        printf("\n");
        free(result);
    }
    return 0;
}