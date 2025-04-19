import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb2 → bb3 [exec=0, time=0.0000s] [mem=127244 bytes] | bb0 → bb2, bb1 [exec=0, time=0.0000s] [mem=127244 bytes] | bb1 → bb3 [exec=0, time=0.0000s] [mem=127244 bytes]

with entry {

    testcases = [  # bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
        ([1, 4, 9, 4, 1], [1, 5, 0, 5, 1]),  # bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
        ([1, 3, 1], [1, 4, 1]),  # bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
        ([4, 7, 2, 5, 5, 2, 7, 4], [4, 7, 2, 6, 6, 2, 7, 4]),  # bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
        ([4, 7, 2, 5, 2, 7, 4], [4, 7, 2, 6, 2, 7, 4]),  # bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
        ([9, 9, 9], [1, 0, 0, 1])  # bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
    ];    
    for (i, (digits, expected)) in enumerate(testcases) {  # bb15 [exec=0, time=0.0000s] [mem=127244 bytes] | bb4 → bb15, bb14, bb5 [exec=0, time=0.0000s] [mem=127244 bytes] | bb3 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]

        digit_list = digits[:];  # bb4 → bb15, bb14, bb5 [exec=0, time=0.0000s] [mem=127244 bytes]
        
        high_mid = (len(digit_list) // 2);  # bb4 → bb15, bb14, bb5 [exec=0, time=0.0000s] [mem=127244 bytes]
        
        low_mid = ((len(digit_list) - 1) // 2);  # bb4 → bb15, bb14, bb5 [exec=0, time=0.0000s] [mem=127244 bytes]
        
        while high_mid < len(digit_list) and low_mid >= 0 {  # bb11 → bb14, bb12 [exec=0, time=0.0000s] [mem=127244 bytes] | bb12 → bb14, bb13 [exec=0, time=0.0000s] [mem=127244 bytes] | bb5 → bb14, bb6 [exec=0, time=0.0000s] [mem=127244 bytes] | bb4 → bb15, bb14, bb5 [exec=0, time=0.0000s] [mem=127244 bytes] | bb13 → bb6 [exec=0, time=0.0000s] [mem=127244 bytes]

            if digit_list[high_mid] == 9 {  # bb6 → bb8, bb7 [exec=0, time=0.0000s] [mem=127244 bytes]

                digit_list[high_mid] = 0;      # bb7 → bb11 [exec=0, time=0.0000s] [mem=127244 bytes]
                digit_list[low_mid] = 0;      # bb7 → bb11 [exec=0, time=0.0000s] [mem=127244 bytes]
                high_mid += 1;      # bb7 → bb11 [exec=0, time=0.0000s] [mem=127244 bytes]
                low_mid -= 1;  # bb7 → bb11 [exec=0, time=0.0000s] [mem=127244 bytes]
            } else {

                digit_list[high_mid] += 1;      # bb8 → bb10, bb9 [exec=0, time=0.0000s] [mem=127244 bytes]
                if low_mid != high_mid {  # bb8 → bb10, bb9 [exec=0, time=0.0000s] [mem=127244 bytes]

                    digit_list[low_mid] += 1;  # bb9 → bb10 [exec=0, time=0.0000s] [mem=127244 bytes]
                }
                
                break ;  # bb10 → bb14 [exec=0, time=0.0000s] [mem=127244 bytes]
            }
        }

        
        smart_assert(digit_list == expected, '') ;      # bb14 → bb4 [exec=0, time=0.0000s] [mem=127244 bytes]
    }
}
