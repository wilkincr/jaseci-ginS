with entry { 
#   Instructions: [SETUP_ANNOTATIONS(None)]
    x:int = 0;
    #   Instructions: [LOAD_CONST(0)]
    y:int = 3;
    #   Instructions: [LOAD_CONST(3)]
    z:int = x + y;
    #   Instructions: [LOAD_NAME(x)]
    for i in range(49) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(y) [no line], BINARY_OP(11) [no line], STORE_NAME(z) [no line], LOAD_NAME(print) [no line], LOAD_NAME(y) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(print) [no line], LOAD_CONST(hello) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
        x = 4 * i + y * (z);
        #   Instructions: [FOR_ITER(130) [no line], STORE_NAME(i) [no line], LOAD_CONST(4), LOAD_NAME(i) [no line], BINARY_OP(5) [no line], LOAD_NAME(y) [no line], LOAD_NAME(z) [no line], BINARY_OP(5) [no line], BINARY_OP(0) [no line], STORE_NAME(x) [no line], LOAD_CONST(2) [no line], BINARY_OP(6) [no line], POP_JUMP_IF_FALSE(124) [no line]]
        if x % 2 {
        #   Instructions: [LOAD_NAME(x)]
            y= 0;
            #   Instructions: [LOAD_CONST(0), STORE_NAME(y) [no line], JUMP_BACKWARD(80) [no line]]
        }
        else {
          y = 4;
          #   Instructions: [LOAD_CONST(4), STORE_NAME(y) [no line], JUMP_BACKWARD(80) [no line]]
        }
    }
    z = x/y;
    #   Instructions: [LOAD_NAME(x)]
    print(y);
    #   Instructions: [PUSH_NULL(None)]
    print("hello");
    #   Instructions: [PUSH_NULL(None)]
}