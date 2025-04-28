import math
from jaclang.runtimelib.gins.smart_assert import smart_assert # Assuming this import works

# BUG: When the total number of elements (x + y) is even, the median calculation
# incorrectly uses minRightX instead of min(minRightX, minRightY). This fails when
# the smaller of the two "right side" elements needed for the median comes from nums2.
def findMedianSortedArrays_buggy(nums1: list[int], nums2: list[int]) -> float:
    x, y = len(nums1), len(nums2)
    if x > y:
        # Ensure nums1 is the shorter array
        nums1, nums2, x, y = nums2, nums1, y, x

    low, high = 0, x
    while low <= high:
        partitionX = (low + high) // 2
        # Calculate partitionY ensuring (partitionX + partitionY) = (x + y + 1) // 2
        partitionY = (x + y + 1) // 2 - partitionX

        # Correctly handle edges using infinity
        maxLeftX = nums1[partitionX - 1] if partitionX != 0 else float('-inf')
        minRightX = nums1[partitionX] if partitionX != x else float('inf')

        maxLeftY = nums2[partitionY - 1] if partitionY != 0 else float('-inf')
        minRightY = nums2[partitionY] if partitionY != y else float('inf')

        # Check if we found the correct partition
        if maxLeftX <= minRightY and maxLeftY <= minRightX:
            # Correct partition found, calculate median
            if (x + y) % 2 == 0:
                # BUGGY LINE: Uses minRightX unconditionally instead of min(minRightX, minRightY)
                return (max(maxLeftX, maxLeftY) + minRightX) / 2.0
                # CORRECT LOGIC:
                # return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2.0
            else:
                # Odd case: median is the max of the left parts
                return float(max(maxLeftX, maxLeftY))
        elif maxLeftX > minRightY:
            # Move partitionX left
            high = partitionX - 1
        else:
            # Move partitionX right
            low = partitionX + 1

    # This part should theoretically not be reached if inputs are sorted arrays
    # raise ValueError("Input arrays are not sorted or other issue")
    return -1.0 # Placeholder for error or unreachable code

# Test cases
testcases_median_buggy = [
    (([1, 3], [2]), 2.0),                 
    (([1, 2], [3, 4]), 2.5),             
    (([0, 0], [0, 0]), 0.0),              
    (([2], []), 2.0),                      
    (([1], [2, 3, 4]), 2.5),              
    (([1, 4], [2, 3]), 2.5),               # << FAILING CASE: Even. Correct median needs max(1,2)=2 and min(4,3)=3 -> (2+3)/2=2.5. Bug uses minRightX=4 -> (2+4)/2=3.0.
    (([5, 6], [1, 2, 3, 4]), 3.5)         # << FAILING CASE: Even. Correct median needs max(?,3)=3 and min(?,4)=4 -> (3+4)/2=3.5. Bug might use wrong minRight. Let's trace.

]

print("--- Testing Buggy Median of Two Sorted Arrays ---")
for (nums1, nums2), expected in testcases_median_buggy:
    # Make copies to avoid modification if the function alters them (it shouldn't here)
    nums1_copy = list(nums1)
    nums2_copy = list(nums2)
    got = findMedianSortedArrays_buggy(nums1_copy, nums2_copy)
    print(f"Input: ({nums1}, {nums2}), Expected: {expected}, Got: {got}")
    smart_assert(got == expected)