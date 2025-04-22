
can power(x: float, n: int) -> float {
#   Instructions: [SETUP_ANNOTATIONS(None)]
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
    #   Instructions: [BUILD_LIST(0)]
    for (base, exp, expected) in testcases {
    #   Instructions: [LOAD_NAME(testcases), END_FOR(None)]
        got:float = power(base, exp);
        #   Instructions: [FOR_ITER(112) [no line], UNPACK_SEQUENCE(3) [no line], STORE_NAME(base) [no line], STORE_NAME(exp) [no line], STORE_NAME(expected) [no line], PUSH_NULL(None), LOAD_NAME(power) [no line], LOAD_NAME(base) [no line], LOAD_NAME(exp) [no line], CALL(2) [no line], STORE_NAME(got) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(got) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(print) [no line], LOAD_NAME(got) [no line], LOAD_NAME(expected) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(46) [no line]]
        print(got == expected);    
        #   Instructions: [PUSH_NULL(None)]
    }
}