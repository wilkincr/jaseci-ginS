from jaclang.runtimelib.gins.smart_assert import smart_assert

testcases = [
    ([[ ], 14], [ ]),
    ([[3, 11, 2, 9, 1, 5], 12], [1, 2, 3, 5, 9, 11]),
    ([[3, 2, 4, 2, 3, 5], 6], [2, 2, 3, 3, 4, 5]),
    ([[1, 3, 4, 6, 4, 2, 9, 1, 2, 9], 10], [1, 1, 2, 2, 3, 4, 4, 6, 9, 9]),
    ([[20, 19, 18, 17, 16, 15, 14, 13, 12, 11], 21], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    ([[20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 30], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
    ([[8, 5, 3, 1, 9, 6, 0, 7, 4, 2, 5], 10], [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
]
x = 1
for i, ((arr, k), expected) in enumerate(testcases):
    counts = [0] * k
    for x in arr:
        counts[x] += 1

    sorted_arr = []
    for i2, count in enumerate(arr): 
        sorted_arr.extend([i2] * count)


    print("Input:", arr)
    print("Expected:", expected)
    print("Got     :", sorted_arr)
    print("Pass    :", sorted_arr == expected)
    print()
y = 1/x
smart_assert(x != 0, "Variable z was used in division")
