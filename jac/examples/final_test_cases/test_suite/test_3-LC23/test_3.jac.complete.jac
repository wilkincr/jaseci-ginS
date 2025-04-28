with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.006 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 12×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 4×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  lists = [] (seen 1×)  lists = [[]] (seen 1×)  solution = [] (seen 2×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 2×)  merged = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)
    test_lists: list[list[list[int]]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [[1,4,5],[1,3,4],[2,6]], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [[]], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [[1,3,5,7],[2,4,6,8]], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [[1],[0]], # BB: 0
        #   Instructions: [LOAD_CONST(1)]
        [[1,2],[1,3,4],[2,6]], # BB: 0
        #   Instructions: [LOAD_CONST(1)]
        [[5,6,7],[1,2,3,8]] # BB: 0
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[list[int]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [1,1,2,3,4,4,5,6], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5,6,7,8], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [0,1], # BB: 0
        #   Instructions: [LOAD_CONST(0)]
        [1,1,2,2,3,4,6], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,5,6,7,8] # BB: 0
        #   Instructions: [BUILD_LIST(0)]
    ];

    for i in range(len(test_lists)) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        lists: list[list[int]] = test_lists[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(340) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_lists), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(lists) [no line], LOAD_CONST(list[list[int]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(lists) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(merged) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(merged) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 20×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 20×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 7×)  lists = [[]] (seen 5×)  lists = [] (seen 4×)  lists = [[1, 3, 5, 7], [2, 4, 6, 8]] (seen 3×)  solution = [] (seen 9×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 7×)  solution = [1, 2, 3, 4, 5, 6, 7, 8] (seen 2×)  merged = [] (seen 11×)  merged = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  merged = [1, 4, 5] (seen 1×)  merged = [1, 4, 5, 1, 3, 4] (seen 1×)  merged = [1, 4, 5, 1, 3, 4, 2, 6] (seen 1×)
        solution: list[int] = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]
        merged: list[int] = []; # BB: 1
        #   Instructions: [BUILD_LIST(0)]

        for lst in lists { # BB: 1
        #   Instructions: [LOAD_NAME(lists), END_FOR(None), LOAD_ATTR(sort) [no line], CALL(0) [no line], POP_TOP(None) [no line], LOAD_NAME(print) [no line], LOAD_NAME(merged) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(190) [no line]]
            merged: list[int] = merged + lst; # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [FOR_ITER(282) [no line], STORE_NAME(lst) [no line], LOAD_NAME(merged), LOAD_NAME(lst) [no line], BINARY_OP(0) [no line], STORE_NAME(merged) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(merged) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(254) [no line]]
            #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 5×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 5×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 3×)  lists = [[]] (seen 1×)  lists = [[1, 3, 5, 7], [2, 4, 6, 8]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  solution = [] (seen 1×)  solution = [1, 2, 3, 4, 5, 6, 7, 8] (seen 1×)  merged = [] (seen 3×)  merged = [1, 4, 5] (seen 1×)  merged = [1, 4, 5, 1, 3, 4] (seen 1×)
        }

        merged.sort(); # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [LOAD_NAME(merged)]
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 6×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 6×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 2×)  lists = [] (seen 2×)  lists = [[]] (seen 2×)  solution = [] (seen 4×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 2×)  merged = [] (seen 4×)  merged = [1, 4, 5, 1, 3, 4, 2, 6] (seen 1×)  merged = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)
        print(merged == solution); # BB: 3
        #   Instructions: [PUSH_NULL(None)]
    }
}