import:py math ;

can is_prime(n: Any) {
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

    testcases:(int, bool) = [(0, False), (1, False), (4, False), (9, False), (17, True), (18, False), (19, True)];    
    for (idx, (inp, expected)) in enumerate(testcases, 1) {

        got:bool = is_prime(inp);    
    }
}
