testcases = [
    (1, [[1]]),
    (2, [[1], [1, 1]]),
    (3, [[1], [1, 1], [1, 2, 1]]),
    (4, [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]),
    (5, [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]])
]

for i, (n, expected) in enumerate(testcases):
    rows = [[1]]
    for r in range(1, n):
        row = []
        for c in range(0, r):
            upleft = rows[r - 1][c - 1] if c > 0 else 0
            upright = rows[r - 1][c] if c < r else 0
            row.append(upleft + upright)
        rows.append(row)

    print("Input:", n)
    print("Expected:", expected)
    print("Got     :", rows)
    print("Pass    :", rows == expected)
    print()
