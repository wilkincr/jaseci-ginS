with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    test_nums: list[list[int]] = [
    #   Instructions: [BUILD_LIST(7)]
        [2,7,9,3,1],
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,1],
        #   Instructions: [BUILD_LIST(0)]
        [2,1,1,2],
        #   Instructions: [BUILD_LIST(0)]
        [5],
        #   Instructions: [LOAD_CONST(5)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [2,1,1,2,1],
        #   Instructions: [BUILD_LIST(0)]
        [10,1,1,10]
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [12, 4, 4, 5, 0, 5, 20];
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_nums)) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        nums: list[int] = test_nums[i];
        #   Instructions: [FOR_ITER(498) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_nums), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], LOAD_CONST(0) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(210) [no line]]
        solution: int = test_solutions[i];
        #   Instructions: [LOAD_NAME(test_solutions)]
        answer: int = 0;
        #   Instructions: [LOAD_CONST(0)]

        if (len(nums) == 0) {
        #   Instructions: [PUSH_NULL(None)]
            answer: int = 0;
            #   Instructions: [LOAD_CONST(0), STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(474) [no line]]
        }
        elif (len(nums) == 1) {
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], LOAD_CONST(1) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(254) [no line]]
            answer:int = nums[0];
            #   Instructions: [LOAD_NAME(nums), LOAD_CONST(0) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(474) [no line]]
        }
        else {
            n: int = len(nums);
            #   Instructions: [PUSH_NULL(None), LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], STORE_NAME(n) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(n) [no line], STORE_SUBSCR(None) [no line], BUILD_LIST(1) [no line], LOAD_NAME(n) [no line], BINARY_OP(5) [no line], STORE_NAME(dp) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(dp) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(dp) [no line], LOAD_CONST(0) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(max) [no line], LOAD_NAME(nums) [no line], LOAD_CONST(0) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_CONST(1) [no line], BINARY_SUBSCR(None) [no line], CALL(2) [no line], LOAD_NAME(dp) [no line], LOAD_CONST(1) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_CONST(2) [no line], LOAD_NAME(n) [no line], CALL(2) [no line], GET_ITER(None) [no line]]
            dp: list[int] = [0]*n;
            #   Instructions: [LOAD_CONST(0)]
            dp[0]:int = nums[0];
            #   Instructions: [LOAD_NAME(nums)]
            dp[1]:int = max(nums[0], nums[1]);
            #   Instructions: [PUSH_NULL(None)]
            for j in range(2, n) {
            #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_CONST(-1) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line]]
                dp[j]:int = max(dp[j-1] + nums[j], dp[j-2] + nums[j]);
                #   Instructions: [FOR_ITER(452) [no line], STORE_NAME(j) [no line], PUSH_NULL(None), LOAD_NAME(max) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(j) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(0) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(j) [no line], LOAD_CONST(2) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(0) [no line], CALL(2) [no line], LOAD_NAME(dp) [no line], LOAD_NAME(j) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(372) [no line]]
            }
            answer:int = dp[-1];
            #   Instructions: [LOAD_NAME(dp)]
        }

        print(answer == solution);
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(answer) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(112) [no line]]
    }
}