import:py math ; # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms

can is_prime(n: Any) { # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
    if n < 2 {

        return False ;
    }
    
    for i in range(2, int(math.sqrt(n))) {
        if (n % i) == 0 {

            return False ;
        }
    }
    
    return True ;
}

with entry {

    testcases:(int, bool) = [(0, False), (1, False), (4, False), (9, False), (17, True), (18, False), (19, True)];     # BB: 3
    for (idx, (inp, expected)) in enumerate(testcases, 1) { # BB: 3

        got:bool = is_prime(inp);     # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
    }
}
