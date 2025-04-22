
can power(x: float, n: int) -> float {
    if n == 0 {

        return 0.0 ;
    }
    
    result:float = 1.0;    
    for _ in range(n) {

        result *= x;
    }
    
    return result ;
}

with entry {

    testcases:(float, int, float) = [(2.0, 3, 8.0), (5.0, 2, 25.0), (10.0, 0, 1.0), (3.0, 1, 3.0)];    
    for (base, exp, expected) in testcases {
        got:float = power(base, exp);
        print(got == expected);    
    }
}
