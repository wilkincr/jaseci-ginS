import math;

with entry {
    testcases_median_buggy: list[tuple[tuple[list[int], list[int]], float]] = [
        (([1, 3], [2]), 2.0),
        (([1, 2], [3, 4]), 2.5),
        (([0, 0], [0, 0]), 0.0),
        (([2], []), 2.0),
        (([1], [2, 3, 4]), 2.5),
        (([1, 4], [2, 3]), 2.5),
        (([5, 6], [1, 2, 3, 4]), 3.5)
    ];
    
    for ((nums1, nums2), expected) in testcases_median_buggy {
        nums1_copy:<>list[int] = <>list(nums1);
        nums2_copy:<>list[int] = <>list(nums2);
        
        # Algorithm directly inline
        x:int = len(nums1_copy);
        y:int = len(nums2_copy);
        if x > y {
            nums1_copy:int = nums2_copy;
            nums2_copy:int = nums1_copy;
            x:int = y;
            y:int = x;
        }
        low:int = 0;
        high:int = x;
        got:float = -1.0;
        
        while low <= high {
            partitionX:int = ((low + high) // 2);
            partitionY:int = ((((x + y) + 1) // 2) - partitionX);
            maxLeftX:float = nums1_copy[(partitionX - 1)] if partitionX != 0 else float('-inf');
            minRightX:float = nums1_copy[partitionX] if partitionX != x else float('inf');
            maxLeftY:float = nums2_copy[(partitionY - 1)] if partitionY != 0 else float('-inf');
            minRightY:float = nums2_copy[partitionY] if partitionY != y else float('inf');
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX {
                if ((x + y) % 2) == 0 {
                    got:float = ((max(maxLeftX, maxLeftY) + minRightX) / 2.0);
                    break;
                } else {
                    got:float = float(max(maxLeftX, maxLeftY));
                    break;
                }
            } elif maxLeftX > minRightY {
                high:int = (partitionX - 1);
            } else {
                low:int = (partitionX + 1);
            }
        }
        
        print(got == expected);
    }
}