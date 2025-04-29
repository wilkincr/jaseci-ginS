with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.004 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 11×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 3×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 2×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)  answer = 4 (seen 1×)
    test_targets: list[int] = [7, 15, 4, 8, 11, 3, 100]; # BB: 0
    #   Instructions: [BUILD_LIST(0)]
    test_nums: list[list[int]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [2,3,1,2,4,3], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,4,4], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,4,4], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,1,1,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,1,1,1] # BB: 0
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [2, 2, 1, 2, 3, 3, 0]; # BB: 0
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_targets)) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        target: int = test_targets[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(478) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_targets), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(target) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(target) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], STORE_NAME(n) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(n) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_NAME(min_length) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_length) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(current_sum) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(current_sum) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(left) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(left) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 24×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 24×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 24×)  target = 7 (seen 14×)  target = 15 (seen 9×)  nums = [2, 3, 1, 2, 4, 3] (seen 14×)  nums = [1, 2, 3, 4, 5] (seen 8×)  solution = 2 (seen 21×)  n = 6 (seen 14×)  n = 5 (seen 6×)  min_length = 4 (seen 8×)  min_length = 7 (seen 6×)  min_length = 6 (seen 5×)  current_sum = 6 (seen 10×)  current_sum = 0 (seen 4×)  current_sum = 2 (seen 1×)  current_sum = 5 (seen 1×)  current_sum = 1 (seen 1×)  left = 3 (seen 8×)  left = 0 (seen 7×)  left = 1 (seen 1×)  left = 2 (seen 1×)  answer = 4 (seen 10×)
        nums: list[int] = test_nums[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_nums)]
        solution: int = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]
        n: int = len(nums); # BB: 1
        #   Instructions: [PUSH_NULL(None)]
        min_length: int = n + 1; # BB: 1
        #   Instructions: [LOAD_NAME(n)]
        current_sum: int = 0; # BB: 1
        #   Instructions: [LOAD_CONST(0)]
        left: int = 0; # BB: 1
        #   Instructions: [LOAD_CONST(0)]

        for right in range(n) { # BB: 1
        #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(n) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(440) [no line]]
            current_sum += nums[right]; # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [FOR_ITER(406) [no line], STORE_NAME(right) [no line], LOAD_NAME(current_sum), LOAD_NAME(nums) [no line], LOAD_NAME(right) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(13) [no line], STORE_NAME(current_sum) [no line], LOAD_NAME(target) [no line], COMPARE_OP(>=) [no line], POP_JUMP_IF_TRUE(326) [no line]]
            #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 21×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 21×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 21×)  target = 7 (seen 15×)  target = 15 (seen 6×)  nums = [2, 3, 1, 2, 4, 3] (seen 15×)  nums = [1, 2, 3, 4, 5] (seen 6×)  solution = 2 (seen 21×)  n = 6 (seen 15×)  n = 5 (seen 6×)  min_length = 7 (seen 8×)  min_length = 4 (seen 7×)  min_length = 6 (seen 6×)  current_sum = 6 (seen 8×)  current_sum = 0 (seen 2×)  current_sum = 2 (seen 2×)  current_sum = 5 (seen 2×)  current_sum = 1 (seen 2×)  left = 0 (seen 14×)  left = 1 (seen 3×)  left = 2 (seen 3×)  left = 3 (seen 1×)  answer = 4 (seen 6×)
            while (current_sum >= target) { # BB: 2
            #   Instructions: [LOAD_NAME(current_sum), LOAD_NAME(current_sum)]
                min_length:int = min(min_length, right - left + 1); # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [PUSH_NULL(None), LOAD_NAME(min) [no line], LOAD_NAME(min_length) [no line], LOAD_NAME(right) [no line], LOAD_NAME(left) [no line], BINARY_OP(10) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], CALL(2) [no line], STORE_NAME(min_length) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_length) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(right) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(23) [no line], STORE_NAME(current_sum) [no line], LOAD_CONST(1) [no line], BINARY_OP(13) [no line], STORE_NAME(left) [no line], LOAD_NAME(target) [no line], COMPARE_OP(>=) [no line], POP_JUMP_IF_FALSE(404) [no line]]
                #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 9×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 9×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 9×)  target = 7 (seen 9×)  nums = [2, 3, 1, 2, 4, 3] (seen 9×)  solution = 2 (seen 9×)  n = 6 (seen 9×)  min_length = 4 (seen 8×)  min_length = 7 (seen 1×)  current_sum = 6 (seen 3×)  current_sum = 8 (seen 2×)  current_sum = 10 (seen 2×)  current_sum = 9 (seen 2×)  left = 0 (seen 3×)  left = 1 (seen 3×)  left = 2 (seen 3×)
                current_sum -= nums[right]; # BB: 4
                #   Instructions: [LOAD_NAME(current_sum)]
                left += 1; # BB: 4
                #   Instructions: [LOAD_NAME(left)]
            }
        }

        if (min_length == n + 1) { # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(min_length)]
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 1×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 1×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)
            answer: int = 0; # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_CONST(0), STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(454) [no line]]
        }
        else {
            answer: int = min_length; # BB: 9 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(min_length), STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line]]
            #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 1×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 1×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)
        }
        print(answer == solution); # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(answer) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(136) [no line]]
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 1×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 1×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)  answer = 4 (seen 1×)
    }
}