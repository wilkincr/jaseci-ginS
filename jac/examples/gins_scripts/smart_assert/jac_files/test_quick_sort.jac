import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

can quick_sort(arr: <>list[int]) -> <>list[int] {
    if len(arr) <= 1 {

        return arr[:] ;
    }

    

    pivot = arr[(len(arr) // 2)];
    

    left = [x for x in arr if x < pivot];    

    right = [x for x in arr if x > pivot];    
    return (quick_sort(left) + quick_sort(right)) ;
}

with entry {

    testcases = [
        ([5, 3, 5, 3, 5], [3, 3, 5, 5, 5]),
        ([3, -1, 2, 0, -5, 2, 3], sorted([3, -1, 2, 0, -5, 2, 3])),
        ([], []),
        ([42], [42]),
        ([2, 2, 2, 2], [2, 2, 2, 2])
    ];    
    for (inp, expected) in testcases {

        got = quick_sort(inp);    
        smart_assert(got == expected) ;
    }
}

