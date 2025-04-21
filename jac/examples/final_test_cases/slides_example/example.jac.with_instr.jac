with entry {
#   Instructions: [LOAD_CONST(0)]
    x = 10;
    #   Instructions: [LOAD_CONST(10)]
    y = 0;    
    #   Instructions: [LOAD_CONST(0)]
    for i in range(10) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(x) [no line], BINARY_OP(11) [no line], STORE_NAME(y) [no line]]
        x = (x - 1);
        #   Instructions: [FOR_ITER(72) [no line], STORE_NAME(i) [no line], LOAD_NAME(x), LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_NAME(x) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(66) [no line]]
        if (x > 0) {
        #   Instructions: [LOAD_NAME(x)]
            y = 0;
            #   Instructions: [LOAD_CONST(0), STORE_NAME(y) [no line], JUMP_BACKWARD(38) [no line]]
        }
    }
    y = (1 / x);
    #   Instructions: [LOAD_CONST(1)]
}