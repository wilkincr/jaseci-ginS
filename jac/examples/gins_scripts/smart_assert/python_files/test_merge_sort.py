from jaclang.runtimelib.gins.smart_assert import smart_assert

def merge_sort(arr: list[int]) -> list[int]:
    # BUG: after merging, only extends the leftover from *left*—
    # any remaining items in right get dropped.
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])  
    return merged

testcases = [
    ([3, 1, 2],       [1, 2, 3]),    #this test case will fail
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([],              []),
    ([1],             [1]),
    ([2, 2, 1],       [1, 2, 2]),
]

for inp, expected in testcases:
    got = merge_sort(inp)
    smart_assert(got == expected)

