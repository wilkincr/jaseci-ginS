with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]

    x:int = 10;
    #   Instructions: [LOAD_CONST(10)]
    y:int = 0;    
    #   Instructions: [LOAD_CONST(0)]
    for i in range(10) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(x) [no line], BINARY_OP(11) [no line], STORE_NAME(y) [no line]]

        x = (x - 1);
        #   Instructions: [FOR_ITER(78) [no line], STORE_NAME(i) [no line], LOAD_NAME(x), LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_NAME(x) [no line], JUMP_BACKWARD(60) [no line]]
    }
    
    y = (1 / x);
    #   Instructions: [LOAD_CONST(1)]
}
