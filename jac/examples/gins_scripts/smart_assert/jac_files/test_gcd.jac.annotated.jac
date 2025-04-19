import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb0 → bb2, bb1 [exec=1, time=0.0004s] | bb1 → bb3 [exec=0, time=0.0000s] | bb2 → bb3 [exec=1, time=0.0029s]

can gcd(a: int, b: int) -> int {  # bb3 → bb4 [exec=1, time=0.0004s]
    
    while b {

        (a, b) = (b, (a % b));
    }

    
    return 0 ;
}

with entry {

    testcases = [(48, 18, 6), (18, 48, 6), (7, 3, 1), (5, 0, 5)];      # bb3 → bb4 [exec=1, time=0.0004s]
    for (x, y, expected) in testcases {  # bb5 [exec=0, time=0.0000s] | bb4 → bb5, bb4 [exec=1, time=0.0000s] | bb3 → bb4 [exec=1, time=0.0004s]

        got = gcd(x, y);      # bb4 → bb5, bb4 [exec=1, time=0.0000s]
        smart_assert(got == expected) ;  # bb4 → bb5, bb4 [exec=1, time=0.0000s]
    }
}
