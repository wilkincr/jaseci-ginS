with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.003 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 3×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 2×)  nums = [1, 2, 0] (seen 1×)  solution = 3 (seen 1×)  n = 3 (seen 1×)  missing = 3 (seen 1×)
    test_nums: list[list[int]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [1,2,0], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [3,4,-1,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [7,8,9,11,12], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,1], # BB: 0
        #   Instructions: [LOAD_CONST(1)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [2], # BB: 0
        #   Instructions: [LOAD_CONST(2)]
        [1,2,3,4,5] # BB: 0
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [3, 2, 1, 2, 1, 1, 6]; # BB: 0
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_nums)) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        nums: list[int] = test_nums[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.002 ms
        #   Instructions: [FOR_ITER(540) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_nums), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], STORE_NAME(n) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(n) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 11×)  test_nums = [[1, 2, 0], [-1, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 12×)  nums = [1, 2, 0] (seen 7×)  nums = [3, 4, -1, 1] (seen 3×)  nums = [-1, 4, 3, 1] (seen 1×)  solution = 3 (seen 7×)  solution = 2 (seen 3×)  n = 3 (seen 7×)  n = 4 (seen 2×)  missing = 3 (seen 5×)  temp = -1 (seen 1×)
        solution: int = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]
        
        n: int = len(nums); # BB: 1
        #   Instructions: [PUSH_NULL(None)]

        for j in range(n) { # BB: 1
        #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_NAME(missing) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(missing) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
            while (nums[j] > 0 and nums[j] <= n and nums[nums[j]-1] != nums[j]) { # BB: 2 Execution frequency: 1 Total execution time: 0.000 ms
            #   Instructions: [FOR_ITER(422) [no line], STORE_NAME(j) [no line], LOAD_NAME(nums), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(224) [no line], LOAD_NAME(nums)]
            #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 4×)  test_nums = [[1, 2, 0], [-1, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_nums = [[1, 2, 0], [-1, 1, 3, 4], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 7×)  nums = [1, 2, 0] (seen 3×)  nums = [-1, 4, 3, 1] (seen 2×)  nums = [3, 4, -1, 1] (seen 1×)  nums = [-1, 1, 3, 4] (seen 1×)  solution = 2 (seen 4×)  solution = 3 (seen 3×)  n = 4 (seen 4×)  n = 3 (seen 3×)  missing = 3 (seen 4×)  temp = -1 (seen 2×)  temp = 1 (seen 1×)
                temp: int = nums[nums[j]-1]; # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [LOAD_NAME(nums), LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(temp) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(temp) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(366) [no line]]
                #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_nums = [[1, 2, 0], [-1, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_nums = [[1, 2, 0], [-1, 1, 3, 4], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_nums = [[1, 2, 0], [3, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_nums = [[1, 2, 0], [-1, 4, 3, 4], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 9×)  nums = [3, 4, -1, 1] (seen 2×)  nums = [-1, 4, 3, 1] (seen 2×)  nums = [-1, 1, 3, 4] (seen 2×)  nums = [3, 4, 3, 1] (seen 1×)  nums = [-1, 4, 3, 4] (seen 1×)  solution = 2 (seen 9×)  n = 4 (seen 9×)  missing = 3 (seen 9×)  temp = -1 (seen 5×)  temp = 1 (seen 3×)
                nums[nums[j]-1]:int = nums[j]; # BB: 8
                #   Instructions: [LOAD_NAME(nums)]
                nums[j]:int = temp; # BB: 8
                #   Instructions: [LOAD_NAME(temp)]
            }
        }

        missing: int = n + 1; # BB: 15 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(n)]
        #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 4×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 4×)  nums = [1, 2, 0] (seen 4×)  solution = 3 (seen 4×)  n = 3 (seen 4×)  missing = 4 (seen 3×)
        for j in range(n) { # BB: 15
        #   Instructions: [PUSH_NULL(None), END_FOR(None)]
            if (nums[j] != j+1) { # BB: 16 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [FOR_ITER(514) [no line], STORE_NAME(j) [no line], LOAD_NAME(nums), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_TRUE(490) [no line]]
            #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 3×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 3×)  nums = [1, 2, 0] (seen 3×)  solution = 3 (seen 3×)  n = 3 (seen 3×)  missing = 4 (seen 3×)
                missing:int = j+1; # BB: 18 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [LOAD_NAME(j), LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_NAME(missing) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(missing) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(516) [no line]]
                #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 2×)  nums = [1, 2, 0] (seen 2×)  solution = 3 (seen 2×)  n = 3 (seen 2×)  missing = 4 (seen 1×)  missing = 3 (seen 1×)
                break; # BB: 18
                #   Instructions: [POP_TOP(None)]
            }
        }

        print(missing == solution); # BB: 20 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(missing) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(112) [no line]]
        #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 1×)  nums = [1, 2, 0] (seen 1×)  solution = 3 (seen 1×)  n = 3 (seen 1×)  missing = 3 (seen 1×)
    }
}