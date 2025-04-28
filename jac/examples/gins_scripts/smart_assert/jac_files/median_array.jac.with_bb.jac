import:py math ; # BB: 0 Execution frequency: 1 Total execution time: 0.001 ms

can findMedianSortedArrays_buggy(nums1: <>list[int], nums2: <>list[int]) -> float { # BB: 3 Execution frequency: 1 Total execution time: 0.002 ms

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
        (([1, 3], [2]), 2.0), # BB: 3
        (([1, 2], [3, 4]), 2.5), # BB: 3
        (([0, 0], [0, 0]), 0.0), # BB: 3
        (([2], []) # BB: 3
        ,2.0), # BB: 3
        (([1], [2, 3, 4]), 2.5), # BB: 3
        (([1, 4], [2, 3]), 2.5), # BB: 3
        (([5, 6], [1, 2, 3, 4]), 3.5) # BB: 3
    ];    

    for ((nums1, nums2), expected) in testcases_median_buggy { # BB: 3

        nums1_copy = <>list(nums1);     # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
        nums2_copy = <>list(nums2);     # BB: 4
        got = findMedianSortedArrays_buggy(nums1_copy, nums2_copy);       # BB: 4
    }
}
