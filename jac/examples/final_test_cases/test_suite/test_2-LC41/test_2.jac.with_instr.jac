with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    test_nums: list[list[int]] = [
    #   Instructions: [BUILD_LIST(7)]
        [1,2,0],
        #   Instructions: [BUILD_LIST(0)]
        [3,4,-1,1],
        #   Instructions: [BUILD_LIST(0)]
        [7,8,9,11,12],
        #   Instructions: [BUILD_LIST(0)]
        [1,1],
        #   Instructions: [LOAD_CONST(1)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [2],
        #   Instructions: [LOAD_CONST(2)]
        [1,2,3,4,5]
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [3, 2, 1, 2, 1, 1, 6];
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_nums)) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        nums: list[int] = test_nums[i];
        #   Instructions: [FOR_ITER(528) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_nums), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], STORE_NAME(n) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(n) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        solution: int = test_solutions[i];
        #   Instructions: [LOAD_NAME(test_solutions)]
        
        n: int = len(nums);
        #   Instructions: [PUSH_NULL(None)]

        for j in range(n) {
        #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_NAME(missing) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(missing) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
            while (nums[j] > 0 and nums[j] <= n and nums[nums[j]-1] != nums[j]) {
            #   Instructions: [FOR_ITER(422) [no line], STORE_NAME(j) [no line], LOAD_NAME(nums), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(224) [no line], LOAD_NAME(nums)]
                temp: int = nums[nums[j]-1];
                #   Instructions: [LOAD_NAME(nums), LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(temp) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(temp) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(j) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_TRUE(366) [no line]]
                nums[nums[j]-1]:int = nums[j];
                #   Instructions: [LOAD_NAME(nums)]
                nums[j]:int = temp;
                #   Instructions: [LOAD_NAME(temp)]
            }
        }

        missing: int = n + 1;
        #   Instructions: [LOAD_NAME(n)]
        for j in range(n) {
        #   Instructions: [PUSH_NULL(None), END_FOR(None)]
            if (nums[j] != j) {
            #   Instructions: [FOR_ITER(502) [no line], STORE_NAME(j) [no line], LOAD_NAME(nums), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(j) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_TRUE(484) [no line]]
                missing:int = j;
                #   Instructions: [LOAD_NAME(j), STORE_NAME(missing) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(missing) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(504) [no line]]
                break;
                #   Instructions: [POP_TOP(None)]
            }
        }

        print(missing == solution);
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(missing) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(112) [no line]]
    }
}