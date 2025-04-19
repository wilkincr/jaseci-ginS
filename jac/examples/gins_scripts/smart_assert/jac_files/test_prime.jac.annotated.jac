import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb0 → bb2, bb1 [exec=1, time=0.0004s] | bb2 → bb3 [exec=1, time=0.0030s] | bb1 → bb3 [exec=0, time=0.0000s]
import:py math ;  # bb3 → bb5, bb4 [exec=1, time=0.0001s] | bb5 → bb6 [exec=1, time=0.0007s] | bb4 → bb6 [exec=0, time=0.0000s]

can is_prime(n: Any) {  # bb6 → bb7 [exec=1, time=0.0004s]
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

    testcases = [(0, False), (1, False), (4, False), (9, False), (17, True), (18, False), (19, True)];      # bb6 → bb7 [exec=1, time=0.0004s]
    for (idx, (inp, expected)) in enumerate(testcases, 1) {  # bb6 → bb7 [exec=1, time=0.0004s] | bb8 [exec=0, time=0.0000s] | bb7 → bb8, bb7 [exec=1, time=0.0000s]

        got = is_prime(inp);      # bb7 → bb8, bb7 [exec=1, time=0.0000s]
        smart_assert(got == expected);  # bb7 → bb8, bb7 [exec=1, time=0.0000s]
    }
}
