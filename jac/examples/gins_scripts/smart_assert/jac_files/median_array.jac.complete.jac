import:py math ; # BB: 0 Execution frequency: 1 Total execution time: 0.001 ms
#   Instructions: [LOAD_CONST(0)]

can findMedianSortedArrays_buggy(nums1: <>list[int], nums2: <>list[int]) -> float { # BB: 3 Execution frequency: 1 Total execution time: 0.002 ms
#   Instructions: [LOAD_CONST(('nums1', 'list[int]', 'nums2', 'list[int]', 'return', 'float')), LOAD_CONST(<code object findMedianSortedArrays_buggy at 0x1081983d0, file "/Users/kazu/Desktop/583gins/jaseci-ginS/jac/examples/gins_scripts/smart_assert/jac_files/median_array.jac", line 3>) [no line], MAKE_FUNCTION(4) [no line], STORE_NAME(findMedianSortedArrays_buggy) [no line], LOAD_CONST(3) [no line], BUILD_LIST(2) [no line], LOAD_CONST(2) [no line], BUILD_LIST(1) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.0) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2) [no line], BUILD_LIST(2) [no line], LOAD_CONST(3) [no line], LOAD_CONST(4) [no line], BUILD_LIST(2) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.5) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(0) [no line], BUILD_LIST(2) [no line], LOAD_CONST(0) [no line], LOAD_CONST(0) [no line], BUILD_LIST(2) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(0.0) [no line], BUILD_TUPLE(2) [no line], BUILD_LIST(1) [no line], BUILD_LIST(0) [no line], BUILD_TUPLE(2) [no line], BUILD_LIST(1) [no line], BUILD_LIST(0) [no line], LOAD_CONST((2, 3, 4)) [no line], LIST_EXTEND(1) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.5) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(4) [no line], BUILD_LIST(2) [no line], LOAD_CONST(2) [no line], LOAD_CONST(3) [no line], BUILD_LIST(2) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(2.5) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(6) [no line], BUILD_LIST(2) [no line], BUILD_LIST(0) [no line], LOAD_CONST((1, 2, 3, 4)) [no line], LIST_EXTEND(1) [no line], BUILD_TUPLE(2) [no line], LOAD_CONST(3.5) [no line], BUILD_TUPLE(2) [no line], STORE_NAME(testcases_median_buggy) [no line], GET_ITER(None) [no line]]

    (x, y) = (len(nums1), len(nums2));    
    if x > y {

        (nums1, nums2, x, y) = (nums2, nums1, y, x);
    }
    
    (low, high) = (0, x);    
    
    while low <= high {

        partitionX = ((low + high) // 2);
        

        partitionY = ((((x + y) + 1) // 2) - partitionX);
        

        maxLeftX = nums1[(partitionX - 1)] if partitionX != 0 else float('-inf');    

        minRightX = nums1[partitionX] if partitionX != x else float('inf');    

        maxLeftY = nums2[(partitionY - 1)] if partitionY != 0 else float('-inf');    

        minRightY = nums2[partitionY] if partitionY != y else float('inf');    
        if maxLeftX <= minRightY
            and maxLeftY <= minRightX {
            if ((x + y) % 2) == 0 {
                return ((max(maxLeftX, maxLeftY) + minRightX) / 2.0) ;
            } else {

                return float(max(maxLeftX, maxLeftY)) ;
            }
        } elif maxLeftX > minRightY {

            high = (partitionX - 1);
        } else {

            low = (partitionX + 1);
        }
    }

    
    return -1.0 ;
}

with entry {

    testcases_median_buggy = [ # BB: 3
    #   Instructions: [BUILD_LIST(7)]
        (([1, 3], [2]), 2.0), # BB: 3
        #   Instructions: [LOAD_CONST(1)]
        (([1, 2], [3, 4]), 2.5), # BB: 3
        #   Instructions: [LOAD_CONST(1)]
        (([0, 0], [0, 0]), 0.0), # BB: 3
        #   Instructions: [LOAD_CONST(0)]
        (([2], []) # BB: 3
        #   Instructions: [LOAD_CONST(2), BUILD_TUPLE(2)]
        ,2.0), # BB: 3
        #   Instructions: [LOAD_CONST(2.0)]
        (([1], [2, 3, 4]), 2.5), # BB: 3
        #   Instructions: [LOAD_CONST(1)]
        (([1, 4], [2, 3]), 2.5), # BB: 3
        #   Instructions: [LOAD_CONST(1)]
        (([5, 6], [1, 2, 3, 4]), 3.5) # BB: 3
        #   Instructions: [LOAD_CONST(5)]
    ];    

    for ((nums1, nums2), expected) in testcases_median_buggy { # BB: 3
    #   Instructions: [LOAD_NAME(testcases_median_buggy), END_FOR(None)]

        nums1_copy = <>list(nums1);     # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
        #   Instructions: [FOR_ITER(302) [no line], UNPACK_SEQUENCE(2) [no line], UNPACK_SEQUENCE(2) [no line], STORE_NAME(nums1) [no line], STORE_NAME(nums2) [no line], STORE_NAME(expected) [no line], PUSH_NULL(None), LOAD_NAME(list) [no line], LOAD_NAME(nums1) [no line], CALL(1) [no line], STORE_NAME(nums1_copy) [no line], LOAD_NAME(list) [no line], LOAD_NAME(nums2) [no line], CALL(1) [no line], STORE_NAME(nums2_copy) [no line], LOAD_NAME(findMedianSortedArrays_buggy) [no line], LOAD_NAME(nums1_copy) [no line], LOAD_NAME(nums2_copy) [no line], CALL(2) [no line], STORE_NAME(got) [no line], JUMP_BACKWARD(232) [no line]]
        nums2_copy = <>list(nums2);     # BB: 4
        #   Instructions: [PUSH_NULL(None)]
        got = findMedianSortedArrays_buggy(nums1_copy, nums2_copy);       # BB: 4
        #   Instructions: [PUSH_NULL(None)]
    }
}
