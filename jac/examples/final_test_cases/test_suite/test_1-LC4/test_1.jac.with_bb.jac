import math; # BB: 0 Execution frequency: 1 Total execution time: 0.001 ms

with entry {
    testcases_median_buggy: list[tuple[tuple[list[int], list[int]], float]] = [ # BB: 3 Execution frequency: 1 Total execution time: 0.005 ms
        (([1, 3], [2]), 2.0), # BB: 3
        (([1, 2], [3, 4]), 2.5), # BB: 3
        (([0, 0], [0, 0]), 0.0), # BB: 3
        (([2], []), 2.0), # BB: 3
        (([1], [2, 3, 4]), 2.5), # BB: 3
        (([1, 4], [2, 3]), 2.5), # BB: 3
        (([5, 6], [1, 2, 3, 4]), 3.5) # BB: 3
    ];
    
    for ((nums1, nums2), expected) in testcases_median_buggy { # BB: 3
        nums1_copy:<>list[int] = <>list(nums1); # BB: 5 Execution frequency: 1 Total execution time: 0.000 ms
        nums2_copy:<>list[int] = <>list(nums2); # BB: 5
        
        # Algorithm directly inline
        x:int = len(nums1_copy); # BB: 5
        y:int = len(nums2_copy); # BB: 5
        if x > y { # BB: 5
            nums1_copy:int = nums2_copy; # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
            nums2_copy:int = nums1_copy; # BB: 6
            x:int = y; # BB: 6
            y:int = x; # BB: 6
        }
        low:int = 0; # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
        high:int = x; # BB: 7
        got:float = -1.0; # BB: 7
        
        while low <= high { # BB: 7
            partitionX:int = ((low + high) // 2); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
            partitionY:int = ((((x + y) + 1) // 2) - partitionX); # BB: 8
            maxLeftX:float = nums1_copy[(partitionX - 1)] if partitionX != 0 else float('-inf'); # BB: 8
            minRightX:float = nums1_copy[partitionX] if partitionX != x else float('inf'); # BB: 11 Execution frequency: 0 Total execution time: 0.000 ms
            maxLeftY:float = nums2_copy[(partitionY - 1)] if partitionY != 0 else float('-inf'); # BB: 14 Execution frequency: 0 Total execution time: 0.000 ms
            minRightY:float = nums2_copy[partitionY] if partitionY != y else float('inf'); # BB: 17 Execution frequency: 0 Total execution time: 0.000 ms
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX { # BB: 20 Execution frequency: 0 Total execution time: 0.000 ms
                if ((x + y) % 2) == 0 { # BB: 22 Execution frequency: 0 Total execution time: 0.000 ms
                    got:float = ((max(maxLeftX, maxLeftY) + minRightX) / 2.0); # BB: 23 Execution frequency: 0 Total execution time: 0.000 ms
                    break; # BB: 23
                } else {
                    got:float = float(max(maxLeftX, maxLeftY)); # BB: 24 Execution frequency: 0 Total execution time: 0.000 ms
                    break; # BB: 24
                }
            } elif maxLeftX > minRightY { # BB: 25 Execution frequency: 0 Total execution time: 0.000 ms
                high:int = (partitionX - 1); # BB: 26 Execution frequency: 0 Total execution time: 0.000 ms
            } else {
                low:int = (partitionX + 1); # BB: 27 Execution frequency: 0 Total execution time: 0.000 ms
            }
        }
        
        print(got == expected); # BB: 30 Execution frequency: 0 Total execution time: 0.000 ms
    }
}