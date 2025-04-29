with entry {
#   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 4×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 1×)  solution = 6 (seen 1×)  solution = 10 (seen 1×)  max_wealth = 12 (seen 1×)  max_wealth = 24 (seen 1×)  wealth = 12 (seen 1×)  wealth = 24 (seen 1×)
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
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 20×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 20×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 8×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 7×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 4×)  solution = 10 (seen 8×)  solution = 6 (seen 7×)  solution = 17 (seen 3×)  max_wealth = 0 (seen 6×)  max_wealth = 12 (seen 4×)  max_wealth = 24 (seen 4×)  max_wealth = 6 (seen 2×)  max_wealth = 16 (seen 1×)  wealth = 12 (seen 5×)  wealth = 24 (seen 5×)  wealth = 0 (seen 3×)  wealth = 6 (seen 2×)  wealth = 16 (seen 1×)
        solution: int = test_solutions[i];
        max_wealth: int = 0;
        wealth: int = 0;

        for customer in accounts {
            for money in customer {
            #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 21×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 21×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 9×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 8×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 4×)  solution = 10 (seen 9×)  solution = 6 (seen 8×)  solution = 17 (seen 4×)  max_wealth = 0 (seen 11×)  max_wealth = 6 (seen 7×)  max_wealth = 16 (seen 3×)  wealth = 6 (seen 4×)  wealth = 0 (seen 3×)  wealth = 1 (seen 2×)  wealth = 16 (seen 2×)  wealth = 3 (seen 1×)
                wealth += money;
                #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 15×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 15×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 6×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 3×)  solution = 6 (seen 6×)  solution = 10 (seen 6×)  solution = 17 (seen 3×)  max_wealth = 0 (seen 8×)  max_wealth = 6 (seen 5×)  max_wealth = 16 (seen 2×)  wealth = 0 (seen 3×)  wealth = 1 (seen 2×)  wealth = 6 (seen 2×)  wealth = 3 (seen 1×)  wealth = 9 (seen 1×)
            }
            if (wealth > max_wealth) {
            #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 6×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 2×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 1×)  solution = 10 (seen 3×)  solution = 6 (seen 2×)  solution = 17 (seen 1×)  max_wealth = 0 (seen 3×)  max_wealth = 6 (seen 2×)  max_wealth = 16 (seen 1×)  wealth = 6 (seen 2×)  wealth = 12 (seen 1×)  wealth = 16 (seen 1×)  wealth = 24 (seen 1×)  wealth = 17 (seen 1×)
                max_wealth:int = wealth;
                #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 6×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 6×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 2×)  accounts = [[2, 8, 7], [7, 1, 3], [1, 9, 5]] (seen 1×)  solution = 10 (seen 3×)  solution = 6 (seen 2×)  solution = 17 (seen 1×)  max_wealth = 0 (seen 3×)  max_wealth = 6 (seen 2×)  max_wealth = 16 (seen 1×)  wealth = 6 (seen 2×)  wealth = 12 (seen 1×)  wealth = 16 (seen 1×)  wealth = 24 (seen 1×)  wealth = 17 (seen 1×)
            }
        }

        print(max_wealth == solution);
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 2×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 2×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 1×)  accounts = [[1, 5], [7, 3], [3, 5]] (seen 1×)  solution = 6 (seen 1×)  solution = 10 (seen 1×)  max_wealth = 12 (seen 1×)  max_wealth = 24 (seen 1×)  wealth = 12 (seen 1×)  wealth = 24 (seen 1×)
    }
}