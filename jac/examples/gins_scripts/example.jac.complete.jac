with entry {  # BB: 0 Execution frequency: 1 Total execution time: 0.001 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
    x:int = 0; # BB: 0
    #   Instructions: [LOAD_CONST(0)]
    y:int = 3; # BB: 0
    #   Instructions: [LOAD_CONST(3)]
    z:int = x + y; # BB: 0
    #   Instructions: [LOAD_NAME(x)]
    for i in range(49) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(y) [no line], BINARY_OP(11) [no line], STORE_NAME(z) [no line], LOAD_NAME(print) [no line], LOAD_NAME(y) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(print) [no line], LOAD_CONST(hello) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
        x = 4 * i + y * (z); # BB: 1 Execution frequency: 50 Total execution time: 0.055 ms
        #   Instructions: [FOR_ITER(130) [no line], STORE_NAME(i) [no line], LOAD_CONST(4), LOAD_NAME(i) [no line], BINARY_OP(5) [no line], LOAD_NAME(y) [no line], LOAD_NAME(z) [no line], BINARY_OP(5) [no line], BINARY_OP(0) [no line], STORE_NAME(x) [no line], LOAD_CONST(2) [no line], BINARY_OP(6) [no line], POP_JUMP_IF_FALSE(124) [no line]]
        #   Variable values in this block:   x = 9 (seen 2×)  x = 4 (seen 2×)  x = 20 (seen 2×)  x = 24 (seen 2×)  x = 28 (seen 2×)  y = 4 (seen 94×)  y = 3 (seen 2×)  y = 0 (seen 2×)  z = 3 (seen 98×)
        if x % 2 { # BB: 1
        #   Instructions: [LOAD_NAME(x)]
            y= 0; # BB: 2 Execution frequency: 1 Total execution time: 0.000 ms
            #   Instructions: [LOAD_CONST(0), STORE_NAME(y) [no line], JUMP_BACKWARD(80) [no line]]
            #   Variable values in this block:   x = 9 (seen 1×)  y = 3 (seen 1×)  z = 3 (seen 1×)
        }
        else {
          y = 4; # BB: 3 Execution frequency: 48 Total execution time: 0.010 ms
          #   Instructions: [LOAD_CONST(4), STORE_NAME(y) [no line], JUMP_BACKWARD(80) [no line]]
          #   Variable values in this block:   x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  x = 28 (seen 1×)  x = 32 (seen 1×)  y = 4 (seen 47×)  y = 0 (seen 1×)  z = 3 (seen 48×)
        }
    }
    z = x/y; # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
    #   Instructions: [LOAD_NAME(x)]
    #   Variable values in this block:   x = 204 (seen 3×)  y = 4 (seen 3×)  z = 51.0 (seen 2×)  z = 3 (seen 1×)
    print(y); # BB: 4
    #   Instructions: [PUSH_NULL(None)]
    print("hello"); # BB: 4
    #   Instructions: [PUSH_NULL(None)]
}