with entry {
#   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 4×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 1×)  solution = 6 (seen 1×)  solution = 10 (seen 1×)  max_wealth = 6 (seen 1×)  max_wealth = 10 (seen 1×)  wealth = 6 (seen 1×)  wealth = 8 (seen 1×)
    test_accounts: list[list[list[int]]] = [
        [[1,2,3],[3,2,1]],
        [[1,5],[7,3],[3,5]],
        [[2,8,7],[7,1,3],[1,9,5]],
        [[5,5],[5,5],[5,5]],
        [[10]],
        [[]],
        [[1,2,3],[4,5,6]]
    ];
    test_solutions: list[int] = [6, 10, 17, 10, 10, 0, 15];

    for i in range(len(test_accounts)) {
        accounts: list[list[int]] = test_accounts[i];
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 18×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 18×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 7×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 6×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 4×)  solution = 10 (seen 7×)  solution = 6 (seen 6×)  solution = 17 (seen 3×)  max_wealth = 6 (seen 6×)  max_wealth = 10 (seen 5×)  max_wealth = 0 (seen 3×)  max_wealth = 17 (seen 1×)  wealth = 6 (seen 7×)  wealth = 8 (seen 5×)  wealth = 10 (seen 1×)  wealth = 17 (seen 1×)
        solution: int = test_solutions[i];
        max_wealth: int = 0;

        for customer in accounts {
            wealth: int = 0;
            #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 27×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 27×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 12×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 10×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 5×)  solution = 10 (seen 12×)  solution = 6 (seen 10×)  solution = 17 (seen 5×)  max_wealth = 0 (seen 14×)  max_wealth = 6 (seen 9×)  max_wealth = 10 (seen 4×)  wealth = 6 (seen 6×)  wealth = 0 (seen 6×)  wealth = 10 (seen 3×)  wealth = 3 (seen 3×)  wealth = 8 (seen 2×)
            for money in customer {
                wealth += money;
                #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 15×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 15×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 6×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 3×)  solution = 6 (seen 6×)  solution = 10 (seen 6×)  solution = 17 (seen 3×)  max_wealth = 0 (seen 8×)  max_wealth = 6 (seen 5×)  max_wealth = 10 (seen 2×)  wealth = 0 (seen 6×)  wealth = 3 (seen 3×)  wealth = 1 (seen 2×)  wealth = 5 (seen 1×)  wealth = 7 (seen 1×)
            }
            if (wealth > max_wealth) {
            #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 6×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 2×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 1×)  solution = 10 (seen 3×)  solution = 6 (seen 2×)  solution = 17 (seen 1×)  max_wealth = 0 (seen 3×)  max_wealth = 6 (seen 2×)  max_wealth = 10 (seen 1×)  wealth = 6 (seen 3×)  wealth = 10 (seen 1×)  wealth = 8 (seen 1×)  wealth = 17 (seen 1×)
                max_wealth:int = wealth;
                #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 4×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 4×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 2×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 1×)  solution = 10 (seen 2×)  solution = 6 (seen 1×)  solution = 17 (seen 1×)  max_wealth = 0 (seen 3×)  max_wealth = 6 (seen 1×)  wealth = 6 (seen 2×)  wealth = 10 (seen 1×)  wealth = 17 (seen 1×)
            }
        }

        print(max_wealth == solution);
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 2×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 2×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 1×)  solution = 6 (seen 1×)  solution = 10 (seen 1×)  max_wealth = 6 (seen 1×)  max_wealth = 10 (seen 1×)  wealth = 6 (seen 1×)  wealth = 8 (seen 1×)
    }
}