import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb0 → bb2, bb1 [exec=1, time=0.0004s] | bb1 → bb3 [exec=0, time=0.0000s] | bb2 → bb3 [exec=1, time=0.0046s]

can insertion_sort(arr: <>list[int]) -> <>list[int] {  # bb3 → bb4 [exec=1, time=0.0007s]

    a = arr[:];
    
    
    for j in range(1, len(a)) {

        key = a[j];    
        i = (j - 1);
        
        while i > 0 and a[i] > key {

            a[(i + 1)] = a[i];    
            i -= 1;
        }
        
        a[(i + 1)] = key;
    }
    
    return a ;
}

with entry {

    testcases = [([2, 1], [1, 2]), ([3, 1, 2], [1, 2, 3]), ([5, 4, 3], [3, 4, 5]), ([1, 2, 3], [1, 2, 3])]  # bb3 → bb4 [exec=1, time=0.0007s]
    ;    
    for (inp, expected) in testcases {  # bb4 → bb5, bb4 [exec=1, time=0.0000s] | bb5 [exec=0, time=0.0000s] | bb3 → bb4 [exec=1, time=0.0007s]

        got = insertion_sort(inp);      # bb4 → bb5, bb4 [exec=1, time=0.0000s]
        smart_assert(got == expected) ;  # bb4 → bb5, bb4 [exec=1, time=0.0000s]
    }
}
