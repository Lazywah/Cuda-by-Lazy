# 測試說明

## 測試版本：
CUDA Toolkit 12.9 Update 1 (12.9.1)

## 測試目標：
測試使用'CPU'和使用'GPU'(優化、未優化、內建)對於並列前綴和的運行速度並比對

## 結果目標：
做出多組數據之運算速度變化，CPU及GPU之運算速度差距及分水嶺(何組數據以下CPU較快、GPU較快)

## 測試細節：

### 測試流程

- 測試主要分組：
    1. 串行 CPU 掃描
    2. 多 Block Blelloch 掃描	(未優化)
    3. 多 Block Blelloch 掃描	(BCAO 優化)
    4. Thrust 函式庫            (Exclusive Scan)
    5. CUB 函式庫	            (Device Scan)
    <br>(額外測試)
    6. 單 Block Blelloch 掃描	(未優化)		(elements <= 1024)(不紀錄數據)
    7. 單 Block Blelloch 掃描	(BCAO 優化)	    (elements <= 1024)(不紀錄數據)

- 數據紀錄格式：
    1. version name | test.* | average | Variance | Standard Deviation

- 版本呼叫函數：
    1. 創造亂數矩陣
    2. 呼叫各版本測試函數
    3. 記錄入list中
    4. 返回Main決定是否匯出數據

- 各測試版本流程簡述：
    1. CPU:
        * 單一串行階段：在 Host (CPU) 上循序執行 O(N) 複雜度的前綴和計算。用作所有並行版本的性能基線。

    2. 多 Block Blelloch 掃描 (未優化 \ BCAO 優化):
        * 三階段 GPU 運算：
            1. Block 內掃描：每個 Block 獨立執行 Blelloch 演算法計算結果，並提取 Block 總和 (Si)。
            2. Offset 計算：對 Si 陣列再次掃描，計算出最終偏移量。
            3. 最終更新：啟動第三個核心將偏移量加回各 Block 結果中。
        <br><br>
        * BCAO 優化：
            1. 在三階段運算的基礎上使用 Bank Conflict Avoidance Offset (BCAO) 機制。
            2. 透過調整共享記憶體存取索引來避免記憶體體衝突的優化技術。

    3. Thrust 函式庫 (Exclusive Scan):
        * 高階自動化：
        * Thrust 自動處理數據傳輸、記憶體管理，並在內部自動選擇並執行針對 GPU 優化的多階段並行掃描核心。

    4. CUB 函式庫 (Device Scan)
        * NVIDIA 最佳化：
        * CUB 是 NVIDIA 提供的原始建構單元，執行經過深度優化、針對硬體特性調整的並行掃描算法，通常在大型數據集上能展現最佳的 GPU 性能。

    5. 單 Block Blelloch 掃描 (未優化 \ BCAO 優化):
        * 單 Block 階段：
            1. 僅使用 一個 Block 啟動 未優化 \ BCAO 優化 的單階段掃描核心。
            2. 這種情況下，無需進行第二階段的偏移計算。