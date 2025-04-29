with entry {
#   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 2×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 1×)
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
        #   Variable values in this block:   test_accounts = [[[1, 2, 3], [3, 2, 1]], [[1, 5], [7, 3], [3, 5]], [[2, 8, 7], [7, 1, 3], [1, 9, 5]], [[5, 5], [5, 5], [5, 5]], [[10]], [[]], [[1, 2, 3], [4, 5, 6]]] (seen 3×)  test_solutions = [6, 10, 17, 10, 10, 0, 15] (seen 3×)  accounts = [[1, 2, 3], [3, 2, 1]] (seen 2×)  solution = 6 (seen 1×)
        solution: int = test_solutions[i];
        max_wealth: int = 0;
        wealth: int = 0;

        for customer in accounts {
            for money in customer {
                wealth += money;
            }
            if (wealth > max_wealth) {
                max_wealth:int = wealth;
            }
        }

        print(max_wealth == solution);
    }
}