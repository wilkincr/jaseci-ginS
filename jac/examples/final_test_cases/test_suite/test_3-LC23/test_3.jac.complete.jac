with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 9×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)
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
        lists: list[list[int]] = test_lists[i]; # BB: 1 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(614) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_lists), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(lists) [no line], LOAD_CONST(list[list[int]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(lists) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(merged) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(merged) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(pointers) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(pointers) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(lists) [no line], CALL(1) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 8×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 8×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 7×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 6×)  merged = [] (seen 5×)  pointers = [] (seen 1×)  pointers = [0] (seen 1×)  pointers = [0, 0] (seen 1×)  pointers = [0, 0, 0] (seen 1×)
        solution: list[int] = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]

        merged: list[int] = []; # BB: 1
        #   Instructions: [BUILD_LIST(0)]

        pointers: list[int] = []; # BB: 1
        #   Instructions: [BUILD_LIST(0)]
        for _ in range(len(lists)) { # BB: 1
        #   Instructions: [PUSH_NULL(None), END_FOR(None)]
            pointers.append(0); # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [FOR_ITER(334) [no line], STORE_NAME(_) [no line], LOAD_NAME(pointers), LOAD_ATTR(append) [no line], LOAD_CONST(0) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(292) [no line]]
            #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 3×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 3×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 3×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  merged = [] (seen 3×)  pointers = [] (seen 1×)  pointers = [0] (seen 1×)  pointers = [0, 0] (seen 1×)
        }

        while True { # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [NOP(None), JUMP_BACKWARD(338)]
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)
            min_val: int = 2147483647; # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_CONST(2147483647), STORE_NAME(min_val) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_val) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(min_idx) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_idx) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(lists) [no line], CALL(1) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
            #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 3×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 3×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 3×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  merged = [] (seen 3×)  pointers = [0, 0, 0] (seen 3×)  min_val = 2147483647 (seen 2×)  min_idx = -1 (seen 1×)
            min_idx: int = -1; # BB: 4
            #   Instructions: [LOAD_CONST(-1)]

            for j in range(len(lists)) { # BB: 4
            #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_CONST(-1) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(528) [no line]]
                if pointers[j] < len(lists[j]) { # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [FOR_ITER(514) [no line], STORE_NAME(j) [no line], LOAD_NAME(pointers), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(lists) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], COMPARE_OP(<) [no line], POP_JUMP_IF_TRUE(436) [no line]]
                #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)  min_val = 2147483647 (seen 1×)  min_idx = -1 (seen 1×)
                    if lists[j][pointers[j]] < min_val { # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
                    #   Instructions: [LOAD_NAME(lists), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(pointers) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(min_val) [no line], COMPARE_OP(<) [no line], POP_JUMP_IF_TRUE(466) [no line]]
                    #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)  min_val = 2147483647 (seen 1×)  min_idx = -1 (seen 1×)
                        min_val:int = lists[j][pointers[j]]; # BB: 9 Execution frequency: 0 Total execution time: 0.000 ms
                        #   Instructions: [LOAD_NAME(lists), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(pointers) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(min_val) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_val) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(min_idx) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_idx) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(394) [no line]]
                        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 2×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 2×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 2×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 2×)  merged = [] (seen 2×)  pointers = [0, 0, 0] (seen 2×)  min_val = 2147483647 (seen 1×)  min_val = 1 (seen 1×)  min_idx = -1 (seen 2×)
                        min_idx:int = j; # BB: 9
                        #   Instructions: [LOAD_NAME(j)]
                    }
                }
            }

            if min_idx == -1 { # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(min_idx)]
                break; # BB: 11 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [JUMP_FORWARD(590)]
            }

            merged.append(min_val); # BB: 12 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(merged), LOAD_ATTR(append) [no line], LOAD_NAME(min_val) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(min_idx) [no line], COPY(2) [no line], COPY(2) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(13) [no line], SWAP(3) [no line], SWAP(2) [no line], STORE_SUBSCR(None) [no line]]
            pointers[min_idx] += 1; # BB: 12
            #   Instructions: [LOAD_NAME(pointers)]
        }

        print(merged == solution); # BB: 13 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(merged) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(190) [no line]]
    }
}