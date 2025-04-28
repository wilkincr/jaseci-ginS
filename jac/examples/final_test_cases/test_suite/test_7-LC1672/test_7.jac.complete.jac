with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.004 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 4×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 1×)  solution = 6 (seen 1×)  solution = 10 (seen 1×)  max_wealth = 6 (seen 1×)  max_wealth = 10 (seen 1×)  wealth = 6 (seen 1×)  wealth = 8 (seen 1×)
    test_accounts: list[list[list[int]]] = [ # BB: 0
    #   Instructions: [BUILD_LIST(7)]
        [[1,2,3],[3,2,1]], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [[1,5],[7,3],[3,5]], # BB: 0
        #   Instructions: [LOAD_CONST(1)]
        [[2,8,7],[7,1,3],[1,9,5]], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [[5,5],[5,5],[5,5]], # BB: 0
        #   Instructions: [LOAD_CONST(5)]
        [[10]], # BB: 0
        #   Instructions: [LOAD_CONST(10)]
        [[]], # BB: 0
        #   Instructions: [BUILD_LIST(0)]
        [[1,2,3],[4,5,6]] # BB: 0
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [6, 10, 17, 10, 10, 0, 15]; # BB: 0
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_accounts)) { # BB: 0
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        accounts: list[list[int]] = test_accounts[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.004 ms
        #   Instructions: [FOR_ITER(336) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_accounts), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(accounts) [no line], LOAD_CONST(list[list[int]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(accounts) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(max_wealth) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(max_wealth) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 18×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 18×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 7×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 6×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 4×)  solution = 10 (seen 7×)  solution = 6 (seen 6×)  solution = 17 (seen 3×)  max_wealth = 6 (seen 6×)  max_wealth = 10 (seen 5×)  max_wealth = 0 (seen 3×)  max_wealth = 17 (seen 1×)  wealth = 6 (seen 7×)  wealth = 8 (seen 5×)  wealth = 10 (seen 1×)  wealth = 17 (seen 1×)
        solution: int = test_solutions[i]; # BB: 1
        #   Instructions: [LOAD_NAME(test_solutions)]
        max_wealth: int = 0; # BB: 1
        #   Instructions: [LOAD_CONST(0)]

        for customer in accounts { # BB: 1
        #   Instructions: [LOAD_NAME(accounts), END_FOR(None), LOAD_NAME(print) [no line], LOAD_NAME(max_wealth) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(174) [no line]]
            wealth: int = 0; # BB: 2 Execution frequency: 1 Total execution time: 0.001 ms
            #   Instructions: [FOR_ITER(310) [no line], STORE_NAME(customer) [no line], LOAD_CONST(0), STORE_NAME(wealth) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(wealth) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
            #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 27×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 27×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 12×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 10×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 5×)  solution = 10 (seen 12×)  solution = 6 (seen 10×)  solution = 17 (seen 5×)  max_wealth = 0 (seen 14×)  max_wealth = 6 (seen 9×)  max_wealth = 10 (seen 4×)  wealth = 6 (seen 6×)  wealth = 0 (seen 6×)  wealth = 10 (seen 3×)  wealth = 3 (seen 3×)  wealth = 8 (seen 2×)
            for money in customer { # BB: 2
            #   Instructions: [LOAD_NAME(customer), END_FOR(None), LOAD_NAME(max_wealth) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(294) [no line]]
                wealth += money; # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
                #   Instructions: [FOR_ITER(280) [no line], STORE_NAME(money) [no line], LOAD_NAME(wealth), LOAD_NAME(money) [no line], BINARY_OP(13) [no line], STORE_NAME(wealth) [no line], JUMP_BACKWARD(262) [no line]]
                #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 15×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 15×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 6×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 3×)  solution = 6 (seen 6×)  solution = 10 (seen 6×)  solution = 17 (seen 3×)  max_wealth = 0 (seen 8×)  max_wealth = 6 (seen 5×)  max_wealth = 10 (seen 2×)  wealth = 0 (seen 6×)  wealth = 3 (seen 3×)  wealth = 1 (seen 2×)  wealth = 5 (seen 1×)  wealth = 7 (seen 1×)
            }
            if (wealth > max_wealth) { # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
            #   Instructions: [LOAD_NAME(wealth)]
            #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 6×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 2×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 1×)  solution = 10 (seen 3×)  solution = 6 (seen 2×)  solution = 17 (seen 1×)  max_wealth = 0 (seen 3×)  max_wealth = 6 (seen 2×)  max_wealth = 10 (seen 1×)  wealth = 6 (seen 3×)  wealth = 10 (seen 1×)  wealth = 8 (seen 1×)  wealth = 17 (seen 1×)
                max_wealth:int = wealth; # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
                #   Instructions: [LOAD_NAME(wealth), STORE_NAME(max_wealth) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(max_wealth) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(238) [no line]]
                #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 4×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 4×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 2×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 1×)  solution = 10 (seen 2×)  solution = 6 (seen 1×)  solution = 17 (seen 1×)  max_wealth = 0 (seen 3×)  max_wealth = 6 (seen 1×)  wealth = 6 (seen 2×)  wealth = 10 (seen 1×)  wealth = 17 (seen 1×)
            }
        }

        print(max_wealth == solution); # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
        #   Instructions: [PUSH_NULL(None)]
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 2×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 2×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 1×)  solution = 6 (seen 1×)  solution = 10 (seen 1×)  max_wealth = 6 (seen 1×)  max_wealth = 10 (seen 1×)  wealth = 6 (seen 1×)  wealth = 8 (seen 1×)
    }
}