with entry { # BB: 0 Transitions: BB0->BB1
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 10×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 2×)  nums = [1, 2, 3, 4] (seen 1×)  solution = [1, 3, 6, 10] (seen 1×)  running_sum = [0, 1, 3, 6] (seen 1×)  total = 10 (seen 1×)
    test_nums: list[list[int]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [1,2,3,4], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,1,1,1,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [3,1,2,10,1], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [0], # BB: 0
        #   Instructions: [LOAD_CONST(0)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [5,10,15], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2] # BB: 0
        #   Instructions: [LOAD_CONST(1)]
    ];
    test_solutions: list[list[int]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [1,3,6,10], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [3,4,6,16,17], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [0], # BB: 0
        #   Instructions: [LOAD_CONST(0)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [5,15,30], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,3] # BB: 0
        #   Instructions: [LOAD_CONST(1)]
    ];

    for i in range(len(test_nums)) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        nums: list[int] = test_nums[i]; # BB: 1 Transitions: BB1->BB4, BB1->BB2
        #   Instructions: [FOR_ITER(300) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_nums), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(running_sum) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(running_sum) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(total) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(total) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 15×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 15×)  nums = [1, 2, 3, 4] (seen 9×)  nums = [1, 1, 1, 1, 1] (seen 5×)  solution = [1, 3, 6, 10] (seen 9×)  solution = [1, 2, 3, 4, 5] (seen 4×)  running_sum = [0, 1, 3, 6] (seen 4×)  running_sum = [] (seen 4×)  running_sum = [0] (seen 2×)  running_sum = [0, 1] (seen 1×)  running_sum = [0, 1, 3] (seen 1×)  total = 10 (seen 5×)  total = 0 (seen 2×)  total = 1 (seen 2×)  total = 3 (seen 1×)  total = 6 (seen 1×)
        solution: list[int] = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]
        running_sum: list[int] = []; # BB: 1
        #   Instructions: [BUILD_LIST(0)]
        total: int = 0; # BB: 1
        #   Instructions: [LOAD_CONST(0)]

        for elt in nums { # BB: 1
        #   Instructions: [LOAD_NAME(nums), END_FOR(None), LOAD_NAME(print) [no line], LOAD_NAME(running_sum) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(144) [no line]]
            running_sum.append(total); # BB: 2 Transitions: BB2->BB3, BB2->BB2
            #   Instructions: [FOR_ITER(274) [no line], STORE_NAME(elt) [no line], LOAD_NAME(running_sum), LOAD_ATTR(append) [no line], LOAD_NAME(total) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(elt) [no line], BINARY_OP(13) [no line], STORE_NAME(total) [no line], JUMP_BACKWARD(222) [no line]]
            #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 11×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 11×)  nums = [1, 2, 3, 4] (seen 8×)  nums = [1, 1, 1, 1, 1] (seen 3×)  solution = [1, 3, 6, 10] (seen 8×)  solution = [1, 2, 3, 4, 5] (seen 3×)  running_sum = [0] (seen 4×)  running_sum = [] (seen 2×)  running_sum = [0, 1] (seen 2×)  running_sum = [0, 1, 3] (seen 2×)  running_sum = [0, 1, 3, 6] (seen 1×)  total = 0 (seen 4×)  total = 1 (seen 3×)  total = 3 (seen 2×)  total = 6 (seen 2×)
            total += elt; # BB: 2
            #   Instructions: [LOAD_NAME(total)]
        }

        print(running_sum == solution); # BB: 3 Transitions: BB3->BB1
        #   Instructions: [PUSH_NULL(None)]
        #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 1×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 1×)  nums = [1, 2, 3, 4] (seen 1×)  solution = [1, 3, 6, 10] (seen 1×)  running_sum = [0, 1, 3, 6] (seen 1×)  total = 10 (seen 1×)
    }
}