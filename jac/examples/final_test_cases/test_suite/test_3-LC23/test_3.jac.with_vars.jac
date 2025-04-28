with entry {
#   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 12×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 4×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  lists = [] (seen 1×)  lists = [[]] (seen 1×)  solution = [] (seen 2×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 2×)  merged = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)
    test_lists: list[list[list[int]]] = [
        [[1,4,5],[1,3,4],[2,6]],
        [],
        [[]],
        [[1,3,5,7],[2,4,6,8]],
        [[1],[0]],
        [[1,2],[1,3,4],[2,6]],
        [[5,6,7],[1,2,3,8]]
    ];
    test_solutions: list[list[int]] = [
        [1,1,2,3,4,4,5,6],
        [],
        [],
        [1,2,3,4,5,6,7,8],
        [0,1],
        [1,1,2,2,3,4,6],
        [1,2,3,5,6,7,8]
    ];

    for i in range(len(test_lists)) {
        lists: list[list[int]] = test_lists[i];
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 20×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 20×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 7×)  lists = [[]] (seen 5×)  lists = [] (seen 4×)  lists = [[1, 3, 5, 7], [2, 4, 6, 8]] (seen 3×)  solution = [] (seen 9×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 7×)  solution = [1, 2, 3, 4, 5, 6, 7, 8] (seen 2×)  merged = [] (seen 11×)  merged = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  merged = [1, 4, 5] (seen 1×)  merged = [1, 4, 5, 1, 3, 4] (seen 1×)  merged = [1, 4, 5, 1, 3, 4, 2, 6] (seen 1×)
        solution: list[int] = test_solutions[i];
        merged: list[int] = [];

        for lst in lists {
            merged: list[int] = merged + lst;
            #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 5×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 5×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 3×)  lists = [[]] (seen 1×)  lists = [[1, 3, 5, 7], [2, 4, 6, 8]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  solution = [] (seen 1×)  solution = [1, 2, 3, 4, 5, 6, 7, 8] (seen 1×)  merged = [] (seen 3×)  merged = [1, 4, 5] (seen 1×)  merged = [1, 4, 5, 1, 3, 4] (seen 1×)
        }

        merged.sort();
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 6×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 6×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 2×)  lists = [] (seen 2×)  lists = [[]] (seen 2×)  solution = [] (seen 4×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 2×)  merged = [] (seen 4×)  merged = [1, 4, 5, 1, 3, 4, 2, 6] (seen 1×)  merged = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)
        print(merged == solution);
    }
}