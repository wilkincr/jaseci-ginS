import math;

with entry {
    testcases_median_buggy: list[tuple[tuple[list[int], list[int]], float]] = [
    #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 2×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = 2.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)  minRightX = 2 (seen 1×)  maxLeftY = 2 (seen 1×)  minRightY = inf (seen 1×)
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
        #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 10×)  nums1_copy = [1, 3] (seen 4×)  nums1_copy = [1, 2] (seen 4×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 5×)  nums2_copy = [3, 4] (seen 3×)  x = 2 (seen 4×)  x = 1 (seen 3×)  y = 1 (seen 5×)  y = 2 (seen 1×)  low = 0 (seen 5×)  high = 1 (seen 5×)  got = 2.0 (seen 5×)  partitionX = 0 (seen 5×)  partitionY = 1 (seen 5×)  maxLeftX = -inf (seen 5×)  minRightX = 2 (seen 5×)  maxLeftY = 2 (seen 5×)  minRightY = inf (seen 5×)
        nums2_copy:<>list[int] = <>list(nums2);
        
        # Algorithm directly inline
        x:int = len(nums1_copy);
        y:int = len(nums2_copy);
        if x > y {
            nums1_copy:int = nums2_copy;
            #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 4×)  nums1_copy = [2] (seen 3×)  nums1_copy = [1, 3] (seen 1×)  nums2_copy = [2] (seen 4×)  x = 2 (seen 3×)  x = 1 (seen 1×)  y = 1 (seen 4×)
            nums2_copy:int = nums1_copy;
            x:int = y;
            y:int = x;
        }
        low:int = 0;
        #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 8×)  nums1_copy = [2] (seen 4×)  nums1_copy = [1, 2] (seen 4×)  nums2_copy = [2] (seen 4×)  nums2_copy = [3, 4] (seen 4×)  x = 1 (seen 4×)  x = 2 (seen 4×)  y = 1 (seen 4×)  y = 2 (seen 4×)  low = 0 (seen 7×)  high = 1 (seen 4×)  high = 2 (seen 2×)  got = 2.0 (seen 3×)  got = -1.0 (seen 2×)  partitionX = 0 (seen 4×)  partitionY = 1 (seen 4×)  maxLeftX = -inf (seen 4×)  minRightX = 2 (seen 4×)  maxLeftY = 2 (seen 4×)  minRightY = inf (seen 4×)
        high:int = x;
        got:float = -1.0;
        
        while low <= high {
            partitionX:int = ((low + high) // 2);
            #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 5×)  nums1_copy = [2] (seen 3×)  nums1_copy = [1, 2] (seen 2×)  nums2_copy = [2] (seen 3×)  nums2_copy = [3, 4] (seen 2×)  x = 1 (seen 3×)  x = 2 (seen 2×)  y = 1 (seen 3×)  y = 2 (seen 2×)  low = 0 (seen 5×)  high = 1 (seen 3×)  high = 2 (seen 2×)  got = -1.0 (seen 5×)  partitionX = 0 (seen 3×)  partitionX = 1 (seen 1×)  partitionY = 1 (seen 3×)  maxLeftX = -inf (seen 2×)  minRightX = 2 (seen 2×)  maxLeftY = 2 (seen 2×)  minRightY = inf (seen 2×)
            partitionY:int = ((((x + y) + 1) // 2) - partitionX);
            maxLeftX:float = nums1_copy[(partitionX - 1)] if partitionX != 0 else float('-inf');
            minRightX:float = nums1_copy[partitionX] if partitionX != x else float('inf');
            #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 1×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = -1.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)
            maxLeftY:float = nums2_copy[(partitionY - 1)] if partitionY != 0 else float('-inf');
            #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 1×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = -1.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)  minRightX = 2 (seen 1×)
            minRightY:float = nums2_copy[partitionY] if partitionY != y else float('inf');
            #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 1×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = -1.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)  minRightX = 2 (seen 1×)  maxLeftY = 2 (seen 1×)
            
            if maxLeftX <= minRightY and maxLeftY <= minRightX {
            #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 1×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = -1.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)  minRightX = 2 (seen 1×)  maxLeftY = 2 (seen 1×)  minRightY = inf (seen 1×)
                if ((x + y) % 2) == 0 {
                #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 1×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = -1.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)  minRightX = 2 (seen 1×)  maxLeftY = 2 (seen 1×)  minRightY = inf (seen 1×)
                    got:float = ((max(maxLeftX, maxLeftY) + minRightX) / 2.0);
                    #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 2×)  nums1_copy = [2] (seen 2×)  nums2_copy = [2] (seen 2×)  x = 1 (seen 2×)  y = 1 (seen 2×)  low = 0 (seen 2×)  high = 1 (seen 2×)  got = -1.0 (seen 1×)  got = 2.0 (seen 1×)  partitionX = 0 (seen 2×)  partitionY = 1 (seen 2×)  maxLeftX = -inf (seen 2×)  minRightX = 2 (seen 2×)  maxLeftY = 2 (seen 2×)  minRightY = inf (seen 2×)
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
        #   Variable values in this block:   testcases_median_buggy = [(([1, 3], [2]), 2.0), (([1, 2], [3, 4]), 2.5), (([0, 0], [0, 0]), 0.0), (([2], []), 2.0), (([1], [2, 3, 4]), 2.5), (([1, 4], [2, 3]), 2.5), (([5, 6], [1, 2, 3, 4]), 3.5)] (seen 1×)  nums1_copy = [2] (seen 1×)  nums2_copy = [2] (seen 1×)  x = 1 (seen 1×)  y = 1 (seen 1×)  low = 0 (seen 1×)  high = 1 (seen 1×)  got = 2.0 (seen 1×)  partitionX = 0 (seen 1×)  partitionY = 1 (seen 1×)  maxLeftX = -inf (seen 1×)  minRightX = 2 (seen 1×)  maxLeftY = 2 (seen 1×)  minRightY = inf (seen 1×)
    }
}