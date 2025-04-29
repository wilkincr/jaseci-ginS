with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.009 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 2×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 1×)
    test_nums: list[list[int]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [2,7,9,3,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [2,1,1,2], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [5], # BB: 0
        #   Instructions: [LOAD_CONST(5)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [2,1,1,2,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [10,1,1,10] # BB: 0
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [12, 4, 4, 5, 0, 5, 20]; # BB: 0
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_nums)) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        nums: list[int] = test_nums[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(498) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_nums), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], LOAD_CONST(0) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(210) [no line]]
        #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 4×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 4×)  nums = [2, 7, 9, 3, 1] (seen 3×)  solution = 12 (seen 2×)  answer = 0 (seen 1×)
        solution: int = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]
        answer: int = 0; # BB: 1
        #   Instructions: [LOAD_CONST(0)]

        if (len(nums) == 0) { # BB: 1
        #   Instructions: [PUSH_NULL(None)]
            answer: int = 0; # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_CONST(0), STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(474) [no line]]
        }
        elif (len(nums) == 1) { # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], LOAD_CONST(1) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(254) [no line]]
        #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 1×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 1×)  nums = [2, 7, 9, 3, 1] (seen 1×)  solution = 12 (seen 1×)  answer = 0 (seen 1×)
            answer:int = nums[0]; # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(nums), LOAD_CONST(0) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(474) [no line]]
        }
        else {
            n: int = len(nums); # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [PUSH_NULL(None), LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], STORE_NAME(n) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(n) [no line], STORE_SUBSCR(None) [no line], BUILD_LIST(1) [no line], LOAD_NAME(n) [no line], BINARY_OP(5) [no line], STORE_NAME(dp) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(dp) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(dp) [no line], LOAD_CONST(0) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(max) [no line], LOAD_NAME(nums) [no line], LOAD_CONST(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_CONST(1) [no line], BINARY_SUBSCR(None) [no line], CALL(2) [no line], LOAD_NAME(dp) [no line], LOAD_CONST(1) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_CONST(2) [no line], LOAD_NAME(n) [no line], CALL(2) [no line], GET_ITER(None) [no line]]
            #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 7×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 7×)  nums = [2, 7, 9, 3, 1] (seen 7×)  solution = 12 (seen 7×)  answer = 0 (seen 7×)  n = 5 (seen 6×)  dp = [0, 0, 0, 0, 0] (seen 1×)  dp = [2, 0, 0, 0, 0] (seen 1×)  dp = [2, 7, 0, 0, 0] (seen 1×)  dp = [2, 7, 16, 0, 0] (seen 1×)  dp = [2, 7, 16, 19, 0] (seen 1×)
            dp: list[int] = [0]*n; # BB: 5
            #   Instructions: [LOAD_CONST(0)]
            dp[0]:int = nums[0]; # BB: 5
            #   Instructions: [LOAD_NAME(nums)]
            dp[1]:int = max(nums[0], nums[1]); # BB: 5
            #   Instructions: [PUSH_NULL(None)]
            for j in range(2, n) { # BB: 5
            #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_CONST(-1) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line]]
                dp[j]:int = max(dp[j-1] + nums[j], dp[j-2] + nums[j]); # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [FOR_ITER(452) [no line], STORE_NAME(j) [no line], PUSH_NULL(None), LOAD_NAME(max) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(0) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(j) [no line], LOAD_CONST(2) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(0) [no line], CALL(2) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(j) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(372) [no line]]
                #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 3×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 3×)  nums = [2, 7, 9, 3, 1] (seen 3×)  solution = 12 (seen 3×)  answer = 0 (seen 3×)  n = 5 (seen 3×)  dp = [2, 7, 0, 0, 0] (seen 1×)  dp = [2, 7, 16, 0, 0] (seen 1×)  dp = [2, 7, 16, 19, 0] (seen 1×)
            }
            answer:int = dp[-1]; # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(dp)]
        }

        print(answer == solution); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(answer) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(112) [no line]]
    }
}