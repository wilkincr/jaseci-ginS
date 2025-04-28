import math;
#   Instructions: [SETUP_ANNOTATIONS(None)]

with entry {
    testcases_median_buggy: list[tuple[tuple[list[int], list[int]], float]] = [
    #   Instructions: [BUILD_LIST(7)]
        (([1, 3], [2]), 2.0),
        #   Instructions: [LOAD_CONST(1), LOAD_CONST(3) [no line], BUILD_LIST(2) [no line], LOAD_CONST(2) [no line], BUILD_LIST(1) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.0) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2) [no line], BUILD_LIST(2) [no line], LOAD_CONST(3) [no line], LOAD_CONST(4) [no line], BUILD_LIST(2) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.5) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(0) [no line], BUILD_LIST(2) [no line], LOAD_CONST(0) [no line], LOAD_CONST(0) [no line], BUILD_LIST(2) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(0.0) [no line], BUILD_TUPLE(2) [no line], BUILD_LIST(1) [no line], BUILD_LIST(0) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.0) [no line], BUILD_TUPLE(2) [no line], BUILD_LIST(1) [no line], BUILD_LIST(0) [no line], LOAD_CONST((2, 3, 4)) [no line], LIST_EXTEND(1) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.5) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(4) [no line], BUILD_LIST(2) [no line], LOAD_CONST(2) [no line], LOAD_CONST(3) [no line], BUILD_LIST(2) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.5) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(6) [no line], BUILD_LIST(2) [no line], BUILD_LIST(0) [no line], LOAD_CONST((1, 2, 3, 4)) [no line], LIST_EXTEND(1) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(3.5) [no line], BUILD_TUPLE(2) [no line], STORE_NAME(testcases_median_buggy) [no line], LOAD_CONST(list[tuple[tuple[list[int], list[int]], float]]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(testcases_median_buggy) [no line], STORE_SUBSCR(None) [no line], GET_ITER(None) [no line]]
        (([1, 2], [3, 4]), 2.5),
        #   Instructions: [LOAD_CONST(1)]
        (([0, 0], [0, 0]), 0.0),
        #   Instructions: [LOAD_CONST(0)]
        (([2], []), 2.0),
        #   Instructions: [LOAD_CONST(2)]
        (([1], [2, 3, 4]), 2.5),
        #   Instructions: [LOAD_CONST(1)]
        (([1, 4], [2, 3]), 2.5),
        #   Instructions: [LOAD_CONST(1)]
        (([5, 6], [1, 2, 3, 4]), 3.5)
        #   Instructions: [LOAD_CONST(5)]
    ];
    
    for ((nums1, nums2), expected) in testcases_median_buggy {
    #   Instructions: [LOAD_NAME(testcases_median_buggy), END_FOR(None)]
        nums1_copy:<>list[int] = <>list(nums1);
        #   Instructions: [FOR_ITER(954) [no line], UNPACK_SEQUENCE(2) [no line], UNPACK_SEQUENCE(2) [no line], STORE_NAME(nums1) [no line], STORE_NAME(nums2) [no line], STORE_NAME(expected) [no line], PUSH_NULL(None), LOAD_NAME(list) [no line], LOAD_NAME(nums1) [no line], CALL(1) [no line], STORE_NAME(nums1_copy) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums1_copy) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(list) [no line], LOAD_NAME(nums2) [no line], CALL(1) [no line], STORE_NAME(nums2_copy) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums2_copy) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums1_copy) [no line], CALL(1) [no line], STORE_NAME(x) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(len) [no line], LOAD_NAME(nums2_copy) [no line], CALL(1) [no line], STORE_NAME(y) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(y) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(y) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(426) [no line]]
        nums2_copy:<>list[int] = <>list(nums2);
        #   Instructions: [PUSH_NULL(None)]
        
        # Algorithm directly inline
        x:int = len(nums1_copy);
        #   Instructions: [PUSH_NULL(None)]
        y:int = len(nums2_copy);
        #   Instructions: [PUSH_NULL(None)]
        if x > y {
        #   Instructions: [LOAD_NAME(x)]
            nums1_copy:int = nums2_copy;
            #   Instructions: [LOAD_NAME(nums2_copy), STORE_NAME(nums1_copy) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums1_copy) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(nums2_copy) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums2_copy) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(x) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(x) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(y) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(y) [no line], STORE_SUBSCR(None) [no line]]
            nums2_copy:int = nums1_copy;
            #   Instructions: [LOAD_NAME(nums1_copy)]
            x:int = y;
            #   Instructions: [LOAD_NAME(y)]
            y:int = x;
            #   Instructions: [LOAD_NAME(x)]
        }
        low:int = 0;
        #   Instructions: [LOAD_CONST(0), STORE_NAME(low) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(low) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(high) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(high) [no line], STORE_SUBSCR(None) [no line], STORE_NAME(got) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(got) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(high) [no line], COMPARE_OP(<=) [no line], POP_JUMP_IF_FALSE(928) [no line]]
        high:int = x;
        #   Instructions: [LOAD_NAME(x)]
        got:float = -1.0;
        #   Instructions: [LOAD_CONST(-1.0)]
        
        while low <= high {
        #   Instructions: [LOAD_NAME(low), LOAD_NAME(low), LOAD_NAME(high) [no line], COMPARE_OP(<=) [no line], POP_JUMP_IF_FALSE(928) [no line]]
            partitionX:int = ((low + high) // 2);
            #   Instructions: [LOAD_NAME(low), LOAD_NAME(high) [no line], BINARY_OP(0) [no line], LOAD_CONST(2) [no line], BINARY_OP(2) [no line], STORE_NAME(partitionX) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(partitionX) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(y) [no line], BINARY_OP(0) [no line], LOAD_CONST(1) [no line], BINARY_OP(0) [no line], LOAD_CONST(2) [no line], BINARY_OP(2) [no line], LOAD_NAME(partitionX) [no line], BINARY_OP(10) [no line], STORE_NAME(partitionY) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(partitionY) [no line], STORE_SUBSCR(None) [no line], LOAD_CONST(0) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(568) [no line]]
            partitionY:int = ((((x + y) + 1) // 2) - partitionX);
            #   Instructions: [LOAD_NAME(x)]
            maxLeftX:float = nums1_copy[(partitionX - 1)] if partitionX != 0 else float('-inf');
            #   Instructions: [LOAD_NAME(partitionX)]
            minRightX:float = nums1_copy[partitionX] if partitionX != x else float('inf');
            #   Instructions: [STORE_NAME(maxLeftX) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(maxLeftX) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(partitionX), LOAD_NAME(x) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(614) [no line]]
            maxLeftY:float = nums2_copy[(partitionY - 1)] if partitionY != 0 else float('-inf');
            #   Instructions: [STORE_NAME(minRightX) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(minRightX) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(partitionY), LOAD_CONST(0) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(666) [no line]]
            minRightY:float = nums2_copy[partitionY] if partitionY != y else float('inf');
            #   Instructions: [STORE_NAME(maxLeftY) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(maxLeftY) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(partitionY), LOAD_NAME(y) [no line], COMPARE_OP(!=) [no line], POP_JUMP_IF_FALSE(712) [no line]]
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX {
            #   Instructions: [STORE_NAME(minRightY) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(minRightY) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(maxLeftX), LOAD_NAME(minRightY) [no line], COMPARE_OP(<=) [no line], POP_JUMP_IF_FALSE(864) [no line]]
                if ((x + y) % 2) == 0 {
                #   Instructions: [LOAD_NAME(x), LOAD_NAME(y) [no line], BINARY_OP(0) [no line], LOAD_CONST(2) [no line], BINARY_OP(6) [no line], LOAD_CONST(0) [no line], COMPARE_OP(==) [no line], POP_JUMP_IF_FALSE(822) [no line]]
                    got:float = ((max(maxLeftX, maxLeftY) + minRightX) / 2.0);
                    #   Instructions: [PUSH_NULL(None), LOAD_NAME(max) [no line], LOAD_NAME(maxLeftX) [no line], LOAD_NAME(maxLeftY) [no line], CALL(2) [no line], LOAD_NAME(minRightX) [no line], BINARY_OP(0) [no line], LOAD_CONST(2.0) [no line], BINARY_OP(11) [no line], STORE_NAME(got) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(got) [no line], STORE_SUBSCR(None) [no line]]
                    break;
                    #   Instructions: [JUMP_FORWARD(928)]
                } else {
                    got:float = float(max(maxLeftX, maxLeftY));
                    #   Instructions: [PUSH_NULL(None), LOAD_NAME(float) [no line], PUSH_NULL(None) [no line], LOAD_NAME(max) [no line], LOAD_NAME(maxLeftX) [no line], LOAD_NAME(maxLeftY) [no line], CALL(2) [no line], CALL(1) [no line], STORE_NAME(got) [no line], LOAD_CONST(float) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(got) [no line], STORE_SUBSCR(None) [no line]]
                    break;
                    #   Instructions: [JUMP_FORWARD(928)]
                }
            } elif maxLeftX > minRightY {
            #   Instructions: [LOAD_NAME(maxLeftX), LOAD_NAME(minRightY) [no line], COMPARE_OP(>) [no line], POP_JUMP_IF_FALSE(896) [no line]]
                high:int = (partitionX - 1);
                #   Instructions: [LOAD_NAME(partitionX), LOAD_CONST(1) [no line], BINARY_OP(10) [no line], STORE_NAME(high) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(high) [no line], STORE_SUBSCR(None) [no line], JUMP_FORWARD(916) [no line]]
            } else {
                low:int = (partitionX + 1);
                #   Instructions: [LOAD_NAME(partitionX), LOAD_CONST(1) [no line], BINARY_OP(0) [no line], STORE_NAME(low) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(low) [no line], STORE_SUBSCR(None) [no line]]
            }
        }
        
        print(got == expected);
        #   Instructions: [PUSH_NULL(None), LOAD_NAME(print) [no line], LOAD_NAME(got) [no line], LOAD_NAME(expected) [no line], COMPARE_OP(==) [no line], CALL(1) [no line], POP_TOP(None) [no line], EXTENDED_ARG(1) [no line], JUMP_BACKWARD(236) [no line]]
    }
}