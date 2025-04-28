from jaclang.runtimelib.gins.smart_assert import smart_assert

def quick_sort(arr: list[int]) -> list[int]:
    # BUG: drops all items equal to the pivot by only recursing on < and > partitions
    if len(arr) <= 1:
        return arr[:]
    pivot = arr[len(arr) // 2]
    left  = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + quick_sort(right)

testcases = [
    ([5, 3, 5, 3, 5], [3, 3, 5, 5, 5]),     # this test case is the bug, where pivot=5, but duplicates of 5 get dropped
    ([3, -1, 2, 0, -5, 2, 3], sorted([3, -1, 2, 0, -5, 2, 3])),
    ([], []),
    ([42], [42]),
    ([2, 2, 2, 2], [2, 2, 2, 2]),
]

for inp, expected in testcases:
    got = quick_sort(inp)
    smart_assert(got == expected)
