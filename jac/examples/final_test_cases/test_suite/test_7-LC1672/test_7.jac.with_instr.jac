with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    test_accounts: list[list[list[int]]] = [
    #   Instructions: [BUILD_LIST(7)]
        [[1,2,3],[3,2,1]],
        #   Instructions: [BUILD_LIST(0)]
        [[1,5],[7,3],[3,5]],
        #   Instructions: [LOAD_CONST(1)]
        [[2,8,7],[7,1,3],[1,9,5]],
        #   Instructions: [BUILD_LIST(0)]
        [[5,5],[5,5],[5,5]],
        #   Instructions: [LOAD_CONST(5)]
        [[10]],
        #   Instructions: [LOAD_CONST(10)]
        [[]],
        #   Instructions: [BUILD_LIST(0)]
        [[1,2,3],[4,5,6]]
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [6, 10, 17, 10, 10, 0, 15];
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_accounts)) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        accounts: list[list[int]] = test_accounts[i];
        #   Instructions: [FOR_ITER(336) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_accounts), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(accounts) [no line], LOAD_CONST(list[list[int]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(accounts) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(max_wealth) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(max_wealth) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
        solution: int = test_solutions[i];
        #   Instructions: [LOAD_NAME(test_solutions)]
        max_wealth: int = 0;
        #   Instructions: [LOAD_CONST(0)]

        for customer in accounts {
        #   Instructions: [LOAD_NAME(accounts), END_FOR(None), LOAD_NAME(print) [no line], LOAD_NAME(max_wealth) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(174) [no line]]
            wealth: int = 0;
            #   Instructions: [FOR_ITER(310) [no line], STORE_NAME(customer) [no line], LOAD_CONST(0), STORE_NAME(wealth) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(wealth) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
            for money in customer {
            #   Instructions: [LOAD_NAME(customer), END_FOR(None), LOAD_NAME(max_wealth) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(294) [no line]]
                wealth += money;
                #   Instructions: [FOR_ITER(280) [no line], STORE_NAME(money) [no line], LOAD_NAME(wealth), LOAD_NAME(money) [no line], BINARY_OP(13) [no line], STORE_NAME(wealth) [no line], JUMP_BACKWARD(262) [no line]]
            }
            if (wealth > max_wealth) {
            #   Instructions: [LOAD_NAME(wealth)]
                max_wealth:int = wealth;
                #   Instructions: [LOAD_NAME(wealth), STORE_NAME(max_wealth) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(max_wealth) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(238) [no line]]
            }
        }

        print(max_wealth == solution);
        #   Instructions: [PUSH_NULL(None)]
    }
}