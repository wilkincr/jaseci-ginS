
can power(x: float, n: int) -> float { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
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

    testcases:(float, int, float) = [(2.0, 3, 8.0), (5.0, 2, 25.0), (10.0, 0, 1.0), (3.0, 1, 3.0)];     # BB: 0
    for (base, exp, expected) in testcases { # BB: 0
        got:float = power(base, exp); # BB: 1 Execution frequency: 1 Total execution time: 0.000 ms
        print(got == expected);     # BB: 1
    }
}