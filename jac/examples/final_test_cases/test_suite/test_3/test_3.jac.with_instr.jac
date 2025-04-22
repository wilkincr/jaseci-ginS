with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    v0:float = 300;
    #   Instructions: [LOAD_CONST(300)]
    c_d:float = 0.001;
    #   Instructions: [LOAD_CONST(0.001)]
    runwayLength:float = 100;
    #   Instructions: [LOAD_CONST(100)]

    tolerance:float = 0.000001;
    #   Instructions: [LOAD_CONST(1e-06)]
    timeStep:float = 0.00001; 
    #   Instructions: [LOAD_CONST(1e-05)]
    bmin:float = 0;
    #   Instructions: [LOAD_CONST(0)]
    bmax:float = v0 * v0 / (2.0 * runwayLength);
    #   Instructions: [LOAD_NAME(v0)]
    bmid:float = (bmin + bmax) / 2.0;
    #   Instructions: [LOAD_NAME(bmin)]
    round:int = 0;
    #   Instructions: [LOAD_CONST(0)]

    while ((bmax - bmin) > tolerance and round < 10) {
    #   Instructions: [LOAD_NAME(bmax), LOAD_NAME(bmax), LOAD_NAME(bmin) [no line], BINARY_OP(10) [no line], LOAD_NAME(tolerance) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(538) [no line]]
        round += 1;
        #   Instructions: [LOAD_NAME(round), LOAD_CONST(1) [no line], BINARY_OP(13) [no line], STORE_NAME(round) [no line], LOAD_NAME(bmax) [no line], BINARY_OP(0) [no line], LOAD_CONST(2.0) [no line], BINARY_OP(11) [no line], STORE_NAME(bmid) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(bmid) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(a_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(a_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(x_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_new) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(x_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x_new) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(460) [no line]]
        bmid:float = (bmin + bmax) / 2.0;
        #   Instructions: [LOAD_NAME(bmin)]
        a_old:float = 0;
        #   Instructions: [LOAD_CONST(0)]
        v_old:float = v0;
        #   Instructions: [LOAD_NAME(v0)]
        x_old:float = 0;
        #   Instructions: [LOAD_CONST(0)]
        v_new:float = 0;
        #   Instructions: [LOAD_CONST(0)]
        x_new:float = 0;
        #   Instructions: [LOAD_CONST(0)]

        while (v_old > 0 and x_old < runwayLength) {
        #   Instructions: [LOAD_NAME(v_old), LOAD_NAME(v_old)]
            a_old:float = - bmid - c_d * v_old * v_old;
            #   Instructions: [LOAD_NAME(bmid), UNARY_NEGATIVE(None) [no line], LOAD_NAME(c_d) [no line], LOAD_NAME(v_old) [no line], BINARY_OP(5) [no line], LOAD_NAME(v_old) [no line], BINARY_OP(5) [no line], BINARY_OP(10) [no line], STORE_NAME(a_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(a_old) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(a_old) [no line], LOAD_NAME(timeStep) [no line], BINARY_OP(5) [no line], BINARY_OP(0) [no line], STORE_NAME(v_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_new) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(v_old) [no line], LOAD_NAME(timeStep) [no line], BINARY_OP(5) [no line], BINARY_OP(0) [no line], STORE_NAME(x_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x_new) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_old) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(460) [no line]]
            v_new:float = v_old + a_old * timeStep;
            #   Instructions: [LOAD_NAME(v_old)]
            x_new:float = x_old + v_old * timeStep;
            #   Instructions: [LOAD_NAME(x_old)]
            v_old:float = v_new;
            #   Instructions: [LOAD_NAME(v_new)]
            v_old:float = x_new;
            #   Instructions: [LOAD_NAME(x_new)]
        }
    
        if (x_old > runwayLength){
        #   Instructions: [LOAD_NAME(x_old), LOAD_NAME(runwayLength) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(486) [no line]]
            bmin:float = bmid;
            #   Instructions: [LOAD_NAME(bmid), STORE_NAME(bmin) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(bmin) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(510) [no line]]
        }

        elif (x_old < runwayLength){
        #   Instructions: [LOAD_NAME(x_old), LOAD_NAME(runwayLength) [no line], COMPARE_OP(<) [no line], POP_JUMP_IF_FALSE(510) [no line]]
            bmax:float = bmid;
            #   Instructions: [LOAD_NAME(bmid), STORE_NAME(bmax) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(bmax) [no line], STORE_SUBSCR(None) [no line]]
        }
    }

  print(f"b_optimal = {bmax}");
  #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_CONST(b_optimal = ) [no line], LOAD_NAME(bmax) [no line], FORMAT_VALUE((None, False)) [no line], BUILD_STRING(2) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
}