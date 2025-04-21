with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
#   Instructions: [LOAD_CONST(0)]
    x = 10; # BB: 0
    #   Instructions: [LOAD_CONST(10)]
    y = 0;     # BB: 0
    #   Instructions: [LOAD_CONST(0)]
    for i in range(10) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(x) [no line], BINARY_OP(11) [no line], STORE_NAME(y) [no line]]
        x = (x - 1); # BB: 1 Execution frequency: 11 Total execution time: 0.003 ms
        #   Instructions: [FOR_ITER(72) [no line], STORE_NAME(i) [no line], LOAD_NAME(x), LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_NAME(x) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(66) [no line]]
        if (x > 0) { # BB: 1
        #   Instructions: [LOAD_NAME(x)]
            y = 0; # BB: 3 Execution frequency: 9 Total execution time: 0.000 ms
            #   Instructions: [LOAD_CONST(0), STORE_NAME(y) [no line], JUMP_BACKWARD(38) [no line]]
        }
    }
    y = (1 / x); # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
    #   Instructions: [LOAD_CONST(1)]
}