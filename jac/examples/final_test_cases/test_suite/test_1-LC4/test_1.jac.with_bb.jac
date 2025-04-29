import math; # BB: 0 Transitions: BB0->BB2, BB0->BB1

with entry {
    testcases_median_buggy: list[tuple[tuple[list[int], list[int]], float]] = [ # BB: 3 Transitions: BB3->BB4
        (([1, 3], [2]), 2.0), # BB: 3
        (([1, 2], [3, 4]), 2.5), # BB: 3
        (([0, 0], [0, 0]), 0.0), # BB: 3
        (([2], []), 2.0), # BB: 3
        (([1], [2, 3, 4]), 2.5), # BB: 3
        (([1, 4], [2, 3]), 2.5), # BB: 3
        (([5, 6], [1, 2, 3, 4]), 3.5) # BB: 3
    ];
    
    for ((nums1, nums2), expected) in testcases_median_buggy { # BB: 3
        nums1_copy:<>list[int] = <>list(nums1); # BB: 5 Transitions: BB5->BB31, BB5->BB7, BB5->BB6
        nums2_copy:<>list[int] = <>list(nums2); # BB: 5
        
        # Algorithm directly inline
        x:int = len(nums1_copy); # BB: 5
        y:int = len(nums2_copy); # BB: 5
        if x > y { # BB: 5
            nums1_copy:int = nums2_copy; # BB: 6 Transitions: BB6->BB7
            nums2_copy:int = nums1_copy; # BB: 6
            x:int = y; # BB: 6
            y:int = x; # BB: 6
        }
        low:int = 0; # BB: 7 Transitions: BB7->BB30, BB7->BB8
        high:int = x; # BB: 7
        got:float = -1.0; # BB: 7
        
        while low <= high { # BB: 7
            partitionX:int = ((low + high) // 2); # BB: 8 Transitions: BB8->BB10, BB8->BB9
            partitionY:int = ((((x + y) + 1) // 2) - partitionX); # BB: 8
            maxLeftX:float = nums1_copy[(partitionX - 1)] if partitionX != 0 else float('-inf'); # BB: 8
            minRightX:float = nums1_copy[partitionX] if partitionX != x else float('inf'); # BB: 11 Transitions: BB11->BB13, BB11->BB12
            maxLeftY:float = nums2_copy[(partitionY - 1)] if partitionY != 0 else float('-inf'); # BB: 14 Transitions: BB14->BB16, BB14->BB15
            minRightY:float = nums2_copy[partitionY] if partitionY != y else float('inf'); # BB: 17 Transitions: BB17->BB19, BB17->BB18
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX { # BB: 20 Transitions: BB20->BB25, BB20->BB21
                if ((x + y) % 2) == 0 { # BB: 22 Transitions: BB22->BB24, BB22->BB23
                    got:float = ((max(maxLeftX, maxLeftY) + minRightX) / 2.0); # BB: 23 Transitions: BB23->BB30
                    break; # BB: 23
                } else {
                    got:float = float(max(maxLeftX, maxLeftY)); # BB: 24 Transitions: BB24->BB30
                    break; # BB: 24
                }
            } elif maxLeftX > minRightY { # BB: 25 Transitions: BB25->BB27, BB25->BB26
                high:int = (partitionX - 1); # BB: 26 Transitions: BB26->BB28
            } else {
                low:int = (partitionX + 1); # BB: 27 Transitions: BB27->BB28
            }
        }
        
        print(got == expected); # BB: 30 Transitions: BB30->BB4
    }
}