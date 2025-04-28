with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    test_targets: list[int] = [7, 15, 4, 8, 11, 3, 100];
    #   Instructions: [BUILD_LIST(0)]
    test_nums: list[list[int]] = [
    #   Instructions: [BUILD_LIST(7)]
        [2,3,1,2,4,3],
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5],
        #   Instructions: [BUILD_LIST(0)]
        [1,4,4],
        #   Instructions: [BUILD_LIST(0)]
        [1,4,4],
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5],
        #   Instructions: [BUILD_LIST(0)]
        [1,1,1,1],
        #   Instructions: [BUILD_LIST(0)]
        [1,1,1,1]
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[int] = [2, 2, 1, 2, 3, 3, 0];
    #   Instructions: [BUILD_LIST(0)]

    for i in range(len(test_targets)) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        target: int = test_targets[i];
        #   Instructions: [FOR_ITER(478) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_targets), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(target) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(target) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums) [no line], CALL(1) [no line], STORE_NAME(n) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(n) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_NAME(min_length) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_length) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(current_sum) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(current_sum) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(left) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(left) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_NAME(n) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        nums: list[int] = test_nums[i];
        #   Instructions: [LOAD_NAME(test_nums)]
        solution: int = test_solutions[i];
        #   Instructions: [LOAD_NAME(test_solutions)]
        n: int = len(nums);
        #   Instructions: [PUSH_NULL(None)]
        min_length: int = n + 1;
        #   Instructions: [LOAD_NAME(n)]
        current_sum: int = 0;
        #   Instructions: [LOAD_CONST(0)]
        left: int = 0;
        #   Instructions: [LOAD_CONST(0)]

        for right in range(n) {
        #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_NAME(n) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(440) [no line]]
            current_sum += nums[right];
            #   Instructions: [FOR_ITER(406) [no line], STORE_NAME(right) [no line], LOAD_NAME(current_sum), LOAD_NAME(nums) [no line], LOAD_NAME(right) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(13) [no line], STORE_NAME(current_sum) [no line], LOAD_NAME(target) [no line], COMPARE_OP(>=) [no line], POP_JUMP_IF_TRUE(326) [no line]]
            while (current_sum >= target) {
            #   Instructions: [LOAD_NAME(current_sum), LOAD_NAME(current_sum)]
                min_length:int = min(min_length, right - left + 1);
                #   Instructions: [PUSH_NULL(None), LOAD_NAME(min) [no line], LOAD_NAME(min_length) [no line], LOAD_NAME(right) [no line], LOAD_NAME(left) [no line], BINARY_OP(10) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], CALL(2) [no line], STORE_NAME(min_length) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_length) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(nums) [no line], LOAD_NAME(left) [no line], BINARY_SUBSCR(None) [no line], BINARY_OP(23) [no line], STORE_NAME(current_sum) [no line], LOAD_CONST(1) [no line], BINARY_OP(13) [no line], STORE_NAME(left) [no line], LOAD_NAME(target) [no line], COMPARE_OP(>=) [no line], POP_JUMP_IF_FALSE(404) [no line]]
                current_sum -= nums[left];
                #   Instructions: [LOAD_NAME(current_sum)]
                left += 1;
                #   Instructions: [LOAD_NAME(left)]
            }
        }

        if (min_length == n + 1) {
        #   Instructions: [LOAD_NAME(min_length)]
            answer: int = 0;
            #   Instructions: [LOAD_CONST(0), STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(454) [no line]]
        }
        else {
            answer: int = min_length;
            #   Instructions: [LOAD_NAME(min_length), STORE_NAME(answer) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(answer) [no line], STORE_SUBSCR(None) [no line]]
        }
        print(answer == solution);
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(answer) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(136) [no line]]
    }
}