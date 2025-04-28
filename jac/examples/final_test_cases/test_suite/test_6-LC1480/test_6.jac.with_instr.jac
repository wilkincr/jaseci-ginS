with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    test_nums: list[list[int]] = [
    #   Instructions: [BUILD_LIST(7)]
        [1,2,3,4],
        #   Instructions: [BUILD_LIST(0)]
        [1,1,1,1,1],
        #   Instructions: [BUILD_LIST(0)]
        [3,1,2,10,1],
        #   Instructions: [BUILD_LIST(0)]
        [0],
        #   Instructions: [LOAD_CONST(0)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [5,10,15],
        #   Instructions: [BUILD_LIST(0)]
        [1,2]
        #   Instructions: [LOAD_CONST(1)]
    ];
    test_solutions: list[list[int]] = [
    #   Instructions: [BUILD_LIST(7)]
        [1,3,6,10],
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5],
        #   Instructions: [BUILD_LIST(0)]
        [3,4,6,16,17],
        #   Instructions: [BUILD_LIST(0)]
        [0],
        #   Instructions: [LOAD_CONST(0)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [5,15,30],
        #   Instructions: [BUILD_LIST(0)]
        [1,3]
        #   Instructions: [LOAD_CONST(1)]
    ];

    for i in range(len(test_nums)) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        nums: list[int] = test_nums[i];
        #   Instructions: [FOR_ITER(300) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_nums), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(running_sum) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(running_sum) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(total) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(total) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
        solution: list[int] = test_solutions[i];
        #   Instructions: [LOAD_NAME(test_solutions)]
        running_sum: list[int] = [];
        #   Instructions: [BUILD_LIST(0)]
        total: int = 0;
        #   Instructions: [LOAD_CONST(0)]

        for elt in nums {
        #   Instructions: [LOAD_NAME(nums), END_FOR(None), LOAD_NAME(print) [no line], LOAD_NAME(running_sum) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(144) [no line]]
            total += elt;
            #   Instructions: [FOR_ITER(274) [no line], STORE_NAME(elt) [no line], LOAD_NAME(total), LOAD_NAME(elt) [no line], BINARY_OP(13) [no line], STORE_NAME(total) [no line], LOAD_ATTR(append) [no line], LOAD_NAME(total) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(222) [no line]]
            running_sum.append(total);
            #   Instructions: [LOAD_NAME(running_sum)]
        }

        print(running_sum == solution);
        #   Instructions: [PUSH_NULL(None)]
    }
}