with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   x = 10 (seen 2×)  x = 9 (seen 1×)  x = 8 (seen 1×)  x = 7 (seen 1×)  x = 6 (seen 1×)  y = 0 (seen 11×)

    x:int = 10; # BB: 0
    #   Instructions: [LOAD_CONST(10)]
    y:int = 0;     # BB: 0
    #   Instructions: [LOAD_CONST(0)]
    for i in range(10) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(x) [no line], BINARY_OP(11) [no line], STORE_NAME(y) [no line]]

        x = (x - 1); # BB: 1 Execution frequency: 1 Total execution time: 0.001 ms
        #   Instructions: [FOR_ITER(78) [no line], STORE_NAME(i) [no line], LOAD_NAME(x), LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_NAME(x) [no line], JUMP_BACKWARD(60) [no line]]
        #   Variable values in this block:   x = 10 (seen 1×)  x = 9 (seen 1×)  x = 8 (seen 1×)  x = 7 (seen 1×)  x = 6 (seen 1×)  y = 0 (seen 10×)
    }
    
    y = (1 / x); # BB: 2 Execution frequency: 1 Total execution time: 0.000 ms
    #   Instructions: [LOAD_CONST(1)]
    #   Variable values in this block:   x = 0 (seen 1×)  y = 0 (seen 1×)
}
