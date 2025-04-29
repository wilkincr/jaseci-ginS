with entry {
#   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 9×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)
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
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 8×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 8×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 7×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 6×)  merged = [] (seen 5×)  pointers = [] (seen 1×)  pointers = [0] (seen 1×)  pointers = [0, 0] (seen 1×)  pointers = [0, 0, 0] (seen 1×)
        solution: list[int] = test_solutions[i];

        merged: list[int] = [];

        pointers: list[int] = [];
        for _ in range(len(lists)) {
            pointers.append(0);
            #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 3×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 3×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 3×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  merged = [] (seen 3×)  pointers = [] (seen 1×)  pointers = [0] (seen 1×)  pointers = [0, 0] (seen 1×)
        }

        while True {
        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)
            min_val: int = 2147483647;
            #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 3×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 3×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 3×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 3×)  merged = [] (seen 3×)  pointers = [0, 0, 0] (seen 3×)  min_val = 2147483647 (seen 2×)  min_idx = -1 (seen 1×)
            min_idx: int = -1;

            for j in range(len(lists)) {
                if pointers[j] < len(lists[j]) {
                #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)  min_val = 2147483647 (seen 1×)  min_idx = -1 (seen 1×)
                    if lists[j][pointers[j]] < min_val {
                    #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)  min_val = 2147483647 (seen 1×)  min_idx = -1 (seen 1×)
                        min_val:int = lists[j][pointers[j]];
                        #   Variable values in this block:   test_lists = [[[1, 4, 5], [1, 3, 4], [2, 6]], [], [[]], [[1, 3, 5, 7], [2, 4, 6, 8]], [[1], [0]], [[1, 2], [1, 3, 4], [2, 6]], [[5, 6, 7], [1, 2, 3, 8]]] (seen 1×)  test_solutions = [[1, 1, 2, 3, 4, 4, 5, 6], [], [], [1, 2, 3, 4, 5, 6, 7, 8], [0, 1], [1, 1, 2, 2, 3, 4, 6], [1, 2, 3, 5, 6, 7, 8]] (seen 1×)  lists = [[1, 4, 5], [1, 3, 4], [2, 6]] (seen 1×)  solution = [1, 1, 2, 3, 4, 4, 5, 6] (seen 1×)  merged = [] (seen 1×)  pointers = [0, 0, 0] (seen 1×)  min_val = 2147483647 (seen 1×)  min_idx = -1 (seen 1×)
                        min_idx:int = j;
                    }
                }
            }

            if min_idx == -1 {
                break;
            }

            merged.append(min_val);
            pointers[min_idx] += 1;
        }

        print(merged == solution);
    }
}