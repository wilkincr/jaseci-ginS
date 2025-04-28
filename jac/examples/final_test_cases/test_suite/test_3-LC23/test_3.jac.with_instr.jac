with entry {
#   Instructions: [SETUP_ANNOTATIONS(None)]
    test_lists: list[list[list[int]]] = [
    #   Instructions: [BUILD_LIST(7)]
        [[1,4,5],[1,3,4],[2,6]],
        #   Instructions: [BUILD_LIST(0)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [[]],
        #   Instructions: [BUILD_LIST(0)]
        [[1,3,5,7],[2,4,6,8]],
        #   Instructions: [BUILD_LIST(0)]
        [[1],[0]],
        #   Instructions: [LOAD_CONST(1)]
        [[1,2],[1,3,4],[2,6]],
        #   Instructions: [LOAD_CONST(1)]
        [[5,6,7],[1,2,3,8]]
        #   Instructions: [BUILD_LIST(0)]
    ];
    test_solutions: list[list[int]] = [
    #   Instructions: [BUILD_LIST(7)]
        [1,1,2,3,4,4,5,6],
        #   Instructions: [BUILD_LIST(0)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [],
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,4,5,6,7,8],
        #   Instructions: [BUILD_LIST(0)]
        [0,1],
        #   Instructions: [LOAD_CONST(0)]
        [1,1,2,2,3,4,6],
        #   Instructions: [BUILD_LIST(0)]
        [1,2,3,5,6,7,8]
        #   Instructions: [BUILD_LIST(0)]
    ];

    for i in range(len(test_lists)) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None)]
        lists: list[list[int]] = test_lists[i];
        #   Instructions: [FOR_ITER(614) [no line], STORE_NAME(i) [no line], LOAD_NAME(test_lists), LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(lists) [no line], LOAD_CONST(list[list[int]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(lists) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(i) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(solution) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(solution) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(merged) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(merged) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(pointers) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(pointers) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(lists) [no line], CALL(1) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
        solution: list[int] = test_solutions[i];
        #   Instructions: [LOAD_NAME(test_solutions)]

        merged: list[int] = [];
        #   Instructions: [BUILD_LIST(0)]

        pointers: list[int] = [];
        #   Instructions: [BUILD_LIST(0)]
        for _ in range(len(lists)) {
        #   Instructions: [PUSH_NULL(None), END_FOR(None)]
            pointers.append(0);
            #   Instructions: [FOR_ITER(334) [no line], STORE_NAME(_) [no line], LOAD_NAME(pointers), LOAD_ATTR(append) [no line], LOAD_CONST(0) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(292) [no line]]
        }

        while True {
        #   Instructions: [NOP(None), JUMP_BACKWARD(338)]
            min_val: int = 2147483647;
            #   Instructions: [LOAD_CONST(2147483647), STORE_NAME(min_val) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_val) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(min_idx) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_idx) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(lists) [no line], CALL(1) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
            min_idx: int = -1;
            #   Instructions: [LOAD_CONST(-1)]

            for j in range(len(lists)) {
            #   Instructions: [PUSH_NULL(None), END_FOR(None), LOAD_CONST(-1) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(528) [no line]]
                if pointers[j] < len(lists[j]) {
                #   Instructions: [FOR_ITER(514) [no line], STORE_NAME(j) [no line], LOAD_NAME(pointers), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], PUSH_NULL(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(lists) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], CALL(1) [no line], COMPARE_OP(<) [no line], POP_JUMP_IF_TRUE(436) [no line]]
                    if lists[j][pointers[j]] < min_val {
                    #   Instructions: [LOAD_NAME(lists), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(pointers) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(min_val) [no line], COMPARE_OP(<) [no line], POP_JUMP_IF_TRUE(466) [no line]]
                        min_val:int = lists[j][pointers[j]];
                        #   Instructions: [LOAD_NAME(lists), LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], LOAD_NAME(pointers) [no line], LOAD_NAME(j) [no line], BINARY_SUBSCR(None) [no line], BINARY_SUBSCR(None) [no line], STORE_NAME(min_val) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_val) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(min_idx) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(min_idx) [no line], STORE_SUBSCR(None) [no line], JUMP_BACKWARD(394) [no line]]
                        min_idx:int = j;
                        #   Instructions: [LOAD_NAME(j)]
                    }
                }
            }

            if min_idx == -1 {
            #   Instructions: [LOAD_NAME(min_idx)]
                break;
                #   Instructions: [JUMP_FORWARD(590)]
            }

            merged.append(min_val);
            #   Instructions: [LOAD_NAME(merged), LOAD_ATTR(append) [no line], LOAD_NAME(min_val) [no line], CALL(1) [no line], POP_TOP(None) [no line], LOAD_NAME(min_idx) [no line], COPY(2) [no line], COPY(2) [no line], BINARY_SUBSCR(None) [no line], LOAD_CONST(1) [no line], BINARY_OP(13) [no line], SWAP(3) [no line], SWAP(2) [no line], STORE_SUBSCR(None) [no line]]
            pointers[min_idx] += 1;
            #   Instructions: [LOAD_NAME(pointers)]
        }

        print(merged == solution);
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(merged) [no line], LOAD_NAME(solution) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], JUMP_BACKWARD(190) [no line]]
    }
}