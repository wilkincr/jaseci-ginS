import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

can insertion_sort(arr: <>list[int]) -> <>list[int] {

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

    testcases = [([2, 1], [1, 2]), ([3, 1, 2], [1, 2, 3]), ([5, 4, 3], [3, 4, 5]), ([1, 2, 3], [1, 2, 3])]
    ;    
    for (inp, expected) in testcases {

        got = insertion_sort(inp);    
        smart_assert(got == expected) ;
    }
}

