with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.003 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   v0 = 300 (seen 9×)  c_d = 0.001 (seen 8×)  runwayLength = 100 (seen 7×)  tolerance = 1e-06 (seen 6×)  timeStep = 1e-05 (seen 5×)  bmin = 0 (seen 4×)  bmax = 450.0 (seen 3×)  bmid = 225.0 (seen 2×)  round = 0 (seen 1×)
    v0:float = 300; # BB: 0
    #   Instructions: [LOAD_CONST(300)]
    c_d:float = 0.001; # BB: 0
    #   Instructions: [LOAD_CONST(0.001)]
    runwayLength:float = 100; # BB: 0
    #   Instructions: [LOAD_CONST(100)]

    tolerance:float = 0.000001; # BB: 0
    #   Instructions: [LOAD_CONST(1e-06)]
    timeStep:float = 0.00001;  # BB: 0
    #   Instructions: [LOAD_CONST(1e-05)]
    bmin:float = 0; # BB: 0
    #   Instructions: [LOAD_CONST(0)]
    bmax:float = v0 * v0 / (2.0 * runwayLength); # BB: 0
    #   Instructions: [LOAD_NAME(v0)]
    bmid:float = (bmin + bmax) / 2.0; # BB: 0
    #   Instructions: [LOAD_NAME(bmin)]
    round:int = 0; # BB: 0
    #   Instructions: [LOAD_CONST(0)]

    while ((bmax - bmin) > tolerance and round < 10) { # BB: 0
    #   Instructions: [LOAD_NAME(bmax), LOAD_NAME(bmax), LOAD_NAME(bmin) [no line], BINARY_OP(10) [no line], LOAD_NAME(tolerance) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(538) [no line]]
        round += 1; # BB: 2 Execution frequency: 1 Total execution time: 0.004 ms
        #   Instructions: [LOAD_NAME(round), LOAD_CONST(1) [no line], BINARY_OP(13) [no line], STORE_NAME(round) [no line], LOAD_NAME(bmax) [no line], BINARY_OP(0) [no line], LOAD_CONST(2.0) [no line], BINARY_OP(11) [no line], STORE_NAME(bmid) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(bmid) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(a_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(a_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(x_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_new) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(x_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x_new) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(460) [no line]]
        #   Variable values in this block:   v0 = 300 (seen 11×)  c_d = 0.001 (seen 11×)  runwayLength = 100 (seen 11×)  tolerance = 1e-06 (seen 11×)  timeStep = 1e-05 (seen 11×)  bmin = 0 (seen 11×)  bmax = 450.0 (seen 11×)  bmid = 225.0 (seen 11×)  round = 1 (seen 10×)  round = 0 (seen 1×)  a_old = 0 (seen 5×)  a_old = -315.0 (seen 1×)  a_old = -225.000000009 (seen 1×)  a_old = -225.0 (seen 1×)  v_old = 300 (seen 4×)  v_old = 0.003 (seen 1×)  v_old = 3.0000000000000004e-08 (seen 1×)  v_old = 3.000000000000001e-13 (seen 1×)  x_old = 0 (seen 6×)  v_new = 0 (seen 2×)  v_new = 299.99685 (seen 1×)  v_new = 0.0007499999999100002 (seen 1×)  v_new = -0.0022499700000000004 (seen 1×)  x_new = 0 (seen 1×)  x_new = 0.003 (seen 1×)  x_new = 3.0000000000000004e-08 (seen 1×)  x_new = 3.000000000000001e-13 (seen 1×)
        bmid:float = (bmin + bmax) / 2.0; # BB: 2
        #   Instructions: [LOAD_NAME(bmin)]
        a_old:float = 0; # BB: 2
        #   Instructions: [LOAD_CONST(0)]
        v_old:float = v0; # BB: 2
        #   Instructions: [LOAD_NAME(v0)]
        x_old:float = 0; # BB: 2
        #   Instructions: [LOAD_CONST(0)]
        v_new:float = 0; # BB: 2
        #   Instructions: [LOAD_CONST(0)]
        x_new:float = 0; # BB: 2
        #   Instructions: [LOAD_CONST(0)]

        while (v_old > 0 and x_old < runwayLength) { # BB: 2
        #   Instructions: [LOAD_NAME(v_old), LOAD_NAME(v_old)]
            a_old:float = - bmid - c_d * v_old * v_old; # BB: 4 Execution frequency: 1 Total execution time: 0.002 ms
            #   Instructions: [LOAD_NAME(bmid), UNARY_NEGATIVE(None) [no line], LOAD_NAME(c_d) [no line], LOAD_NAME(v_old) [no line], BINARY_OP(5) [no line], LOAD_NAME(v_old) [no line], BINARY_OP(5) [no line], BINARY_OP(10) [no line], STORE_NAME(a_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(a_old) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(a_old) [no line], LOAD_NAME(timeStep) [no line], BINARY_OP(5) [no line], BINARY_OP(0) [no line], STORE_NAME(v_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_new) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(v_old) [no line], LOAD_NAME(timeStep) [no line], BINARY_OP(5) [no line], BINARY_OP(0) [no line], STORE_NAME(x_new) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x_new) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_old) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(v_old) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(v_old) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(460) [no line]]
            #   Variable values in this block:   v0 = 300 (seen 16×)  c_d = 0.001 (seen 16×)  runwayLength = 100 (seen 16×)  tolerance = 1e-06 (seen 16×)  timeStep = 1e-05 (seen 16×)  bmin = 0 (seen 16×)  bmax = 450.0 (seen 16×)  bmid = 225.0 (seen 16×)  round = 1 (seen 16×)  a_old = -315.0 (seen 5×)  a_old = -225.000000009 (seen 5×)  a_old = -225.0 (seen 5×)  a_old = 0 (seen 1×)  v_old = 300 (seen 4×)  v_old = 0.003 (seen 4×)  v_old = 3.0000000000000004e-08 (seen 4×)  v_old = 3.000000000000001e-13 (seen 1×)  v_old = 299.99685 (seen 1×)  x_old = 0 (seen 16×)  v_new = 299.99685 (seen 5×)  v_new = 0.0007499999999100002 (seen 5×)  v_new = -0.0022499700000000004 (seen 4×)  v_new = 0 (seen 2×)  x_new = 0.003 (seen 5×)  x_new = 3.0000000000000004e-08 (seen 5×)  x_new = 0 (seen 3×)  x_new = 3.000000000000001e-13 (seen 3×)
            v_new:float = v_old + a_old * timeStep; # BB: 4
            #   Instructions: [LOAD_NAME(v_old)]
            x_new:float = x_old + v_old * timeStep; # BB: 4
            #   Instructions: [LOAD_NAME(x_old)]
            v_old:float = v_new; # BB: 4
            #   Instructions: [LOAD_NAME(v_new)]
            v_old:float = x_new; # BB: 4
            #   Instructions: [LOAD_NAME(x_new)]
        }
    
        if (x_old > runwayLength){ # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(x_old), LOAD_NAME(runwayLength) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(486) [no line]]
            bmin:float = bmid; # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(bmid), STORE_NAME(bmin) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(bmin) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(510) [no line]]
        }

        elif (x_old < runwayLength){ # BB: 9 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(x_old), LOAD_NAME(runwayLength) [no line], COMPARE_OP(<) [no line], POP_JUMP_IF_FALSE(510) [no line]]
            bmax:float = bmid; # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(bmid), STORE_NAME(bmax) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(bmax) [no line], STORE_SUBSCR(None) [no line]]
        }
    }

  print(f"b_optimal = {bmax}"); # BB: 14 Execution frequency: 0 Total execution time: 0.000 ms
  #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_CONST(b_optimal = ) [no line], LOAD_NAME(bmax) [no line], FORMAT_VALUE((None, False)) [no line], BUILD_STRING(2) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
}