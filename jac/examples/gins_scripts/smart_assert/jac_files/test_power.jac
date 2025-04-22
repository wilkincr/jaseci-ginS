import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

can power(x: float, n: int) -> float {
    if n == 0 {

        return 0.0 ;
    }
    
    result = 1.0;    
    for _ in range(n) {

        result *= x;
    }
    
    return result ;
}

with entry {

    testcases = [(2.0, 3, 8.0), (5.0, 2, 25.0), (10.0, 0, 1.0), (3.0, 1, 3.0)];    
    for (base, exp, expected) in testcases {

        got = power(base, exp);    
        smart_assert(got == expected) ;
    }
}

