with entry {  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
    testcases = [  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[], 14], []),  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[3, 11, 2, 9, 1, 5], 12], [1, 2, 3, 5, 9, 11]),  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[3, 2, 4, 2, 3, 5], 6], [2, 2, 3, 3, 4, 5]),  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[1, 3, 4, 6, 4, 2, 9, 1, 2, 9], 10], [1, 1, 2, 2, 3, 4, 4, 6, 9, 9]),  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[20, 19, 18, 17, 16, 15, 14, 13, 12, 11], 21], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 30], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
        ([[8, 5, 3, 1, 9, 6, 0, 7, 4, 2, 5], 10], [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes]
    ];    
    for (i, ((arr, k), expected)) in enumerate(testcases) {  # bb0 → bb1 [exec=1, time=0.0027s] [mem=223884 bytes] | bb6 [exec=0, time=0.0000s] [mem=223884 bytes] | bb1 → bb6, bb2 [exec=8, time=0.0051s] [mem=223884 bytes]

        counts = ([0] * k);  # bb1 → bb6, bb2 [exec=8, time=0.0051s] [mem=223884 bytes]
        
        for x in arr {  # bb3 → bb4 [exec=7, time=0.0018s] [mem=223884 bytes] | bb2 → bb3, bb2 [exec=7, time=0.0903s] [mem=223884 bytes] | bb1 → bb6, bb2 [exec=8, time=0.0051s] [mem=223884 bytes]

            counts[x] += 1;  # bb2 → bb3, bb2 [exec=7, time=0.0903s] [mem=223884 bytes]
        }
        
        sorted_arr = [];  # bb3 → bb4 [exec=7, time=0.0018s] [mem=223884 bytes]
        
        for (i2, count) in enumerate(arr) {  # bb5 → bb1 [exec=7, time=0.0003s] [mem=223884 bytes] | bb3 → bb4 [exec=7, time=0.0018s] [mem=223884 bytes] | bb4 → bb5, bb4 [exec=7, time=0.1439s] [mem=223884 bytes]

            sorted_arr.extend(([i2] * count)) ;  # bb4 → bb5, bb4 [exec=7, time=0.1439s] [mem=223884 bytes]
        }

    }
}
