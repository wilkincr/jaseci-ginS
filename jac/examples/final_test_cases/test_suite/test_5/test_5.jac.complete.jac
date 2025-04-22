

with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.003 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   s1 = ABCBDAB (seen 12×)  s2 = BDCABA (seen 11×)  m = 7 (seen 10×)  n = 6 (seen 9×)
    s1:str = 'ABCBDAB'; # BB: 0
    #   Instructions: [LOAD_CONST(ABCBDAB)]
    s2:str = 'BDCABA'; # BB: 0
    #   Instructions: [LOAD_CONST(BDCABA)]
    m:int = len(s1); # BB: 0
    #   Instructions: [PUSH_NULL(None)]
    n:int = len(s2); # BB: 0
    #   Instructions: [PUSH_NULL(None)]
    dp:list[list[int]] = []; # BB: 0
    #   Instructions: [BUILD_LIST(0)]
    for i in range(m + 1) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(range) [no line], LOAD_NAME(m) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        row:list[int] = []; # BB: 1 Execution frequency: 2 Total execution time: 0.001 ms
        #   Instructions: [FOR_ITER(254) [no line], STORE_NAME(i) [no line], BUILD_LIST(0), STORE_NAME(row) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(row) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   s1 = ABCBDAB (seen 64×)  s2 = BDCABA (seen 64×)  m = 7 (seen 64×)  n = 6 (seen 64×)
        for j in range(n + 1) { # BB: 1
        #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_ATTR(append) [no line], LOAD_NAME(row) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(132) [no line]]
            row.append(0); # BB: 2 Execution frequency: 2 Total execution time: 0.003 ms
            #   Instructions: [FOR_ITER(216) [no line], STORE_NAME(j) [no line], LOAD_NAME(row), LOAD_ATTR(append) [no line], LOAD_CONST(0) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(174) [no line]]
            #   Variable values in this block:   s1 = ABCBDAB (seen 49×)  s2 = BDCABA (seen 49×)  m = 7 (seen 49×)  n = 6 (seen 49×)
        }
        dp.append(row); # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(dp)]
        #   Variable values in this block:   s1 = ABCBDAB (seen 7×)  s2 = BDCABA (seen 7×)  m = 7 (seen 7×)  n = 6 (seen 7×)
    }
  
   
    for i in range(m) { # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
    #   Instructions: [PUSH_NULL(None), END_FOR(None), STORE_NAME(i) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(i) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(j) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(j) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(seq) [no line], LOAD_CONST(list[str]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(seq) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(710) [no line]]
        for j in range(n) { # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(454) [no line], STORE_NAME(i) [no line], PUSH_NULL(None), LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line], END_FOR(None), JUMP_BACKWARD(272) [no line]]
            if s1[i] == s2[j] { # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [FOR_ITER(450) [no line], STORE_NAME(j) [no line], LOAD_NAME(s1), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(s2) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(370) [no line]]
                dp[(i + 1)][(j + 1)]:int = (dp[i][j] + 1); # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [LOAD_NAME(dp), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(294) [no line]]
            } else {
                dp[(i + 1)][(j + 1)]:int = max(dp[i][(j + 1)], dp[(i + 1)][j]); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [PUSH_NULL(None), LOAD_NAME(max) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], CALL(2) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(294) [no line]]
            }
        }
    }
    i:int = m; # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
    #   Instructions: [LOAD_NAME(m)]
    j:int = n; # BB: 10
    #   Instructions: [LOAD_NAME(n)]
   
    seq:list[str] = []; # BB: 10
    #   Instructions: [BUILD_LIST(0)]
   
   
    while i > 0 and j > 0 { # BB: 10
    #   Instructions: [LOAD_NAME(i), LOAD_NAME(i), LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(710) [no line]]
        if s1[(i - 1)] == s2[(j - 1)] { # BB: 12 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(s1), LOAD_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(s2) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(620) [no line]]
            seq.append(s1[(i - 1)]); # BB: 13 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(seq), LOAD_ATTR(append) [no line], LOAD_NAME(s1) [no line], LOAD_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(23) [no line], STORE_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(23) [no line], STORE_NAME(j) [no line], JUMP_FORWARD(688) [no line]]
           
            i -= 1;     # BB: 13
            #   Instructions: [LOAD_NAME(i)]
            j -= 1; # BB: 13
            #   Instructions: [LOAD_NAME(j)]
        } elif dp[(i - 1)][j] >= dp[i][(j - 1)] { # BB: 14 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(dp), LOAD_NAME(i) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], COMPARE_OP(>=) [no line], POP_JUMP_IF_FALSE(678) [no line]]
            i -= 1; # BB: 15 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(i), LOAD_CONST(1) [no line], BINARY_OP(23) [no line], STORE_NAME(i) [no line], JUMP_FORWARD(688) [no line]]
        } else {
            j -= 1; # BB: 16 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(j), LOAD_CONST(1) [no line], BINARY_OP(23) [no line], STORE_NAME(j) [no line]]
        }
    }
   
    print(reversed(seq)); # BB: 20 Execution frequency: 0 Total execution time: 0.000 ms
    #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], PUSH_NULL(None) [no line], LOAD_NAME(reversed) [no line], LOAD_NAME(seq) [no line], CALL(1) [no line], CALL(1) [no line], POP_TOP(None) [no line]]
}