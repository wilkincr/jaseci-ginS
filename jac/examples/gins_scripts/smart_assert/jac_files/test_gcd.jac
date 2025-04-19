import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

can gcd(a: int, b: int) -> int {
    
    while b {

        (a, b) = (b, (a % b));
    }

    
    return 0 ;
}

with entry {

    testcases = [(48, 18, 6), (18, 48, 6), (7, 3, 1), (5, 0, 5)];    
    for (x, y, expected) in testcases {

        got = gcd(x, y);    
        smart_assert(got == expected) ;
    }
}

