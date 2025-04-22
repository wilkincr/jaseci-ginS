import:py math ; # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]

can is_prime(n: Any) { # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
#   Instructions: [LOAD_CONST(('n', 'Any', 'return', 'None')), LOAD_CONST(<code object is_prime at 0x7f9596c78b70, file "/root/jaseci-ginS/jac/examples/final_test_cases/test_suite/test_6/test_6.jac", line 3>) [no line], MAKE_FUNCTION(4) [no line], STORE_NAME(is_prime) [no line], LOAD_CONST(((0, False), (1, False), (4, False), (9, False), (17, True), (18, False), (19, True))) [no line], LIST_EXTEND(1) [no line], STORE_NAME(testcases) [no line], LOAD_CONST((int, bool)) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(testcases) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(enumerate) [no line], LOAD_NAME(testcases) [no line], LOAD_CONST(1) [no line], CALL(2) [no line], GET_ITER(None) [no line]]
#   Variable values in this block:   got = True (seen 4×)  got = False (seen 3×)
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
    #   Instructions: [BUILD_LIST(0)]
    for (idx, (inp, expected)) in enumerate(testcases, 1) { # BB: 3
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]

        got:bool = is_prime(inp);     # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(192) [no line], UNPACK_SEQUENCE(2) [no line], STORE_NAME(idx) [no line], UNPACK_SEQUENCE(2) [no line], STORE_NAME(inp) [no line], STORE_NAME(expected) [no line], PUSH_NULL(None), LOAD_NAME(is_prime) [no line], LOAD_NAME(inp) [no line], CALL(1) [no line], STORE_NAME(got) [no line], LOAD_CONST(bool) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(got) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(146) [no line]]
        #   Variable values in this block:   got = False (seen 4×)  got = True (seen 3×)
    }
}
