with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
    test_accounts: list[list[list[int]]] = [ # BB: 0
        [[1,2,3],[3,2,1]], # BB: 0
        [[1,5],[7,3],[3,5]], # BB: 0
        [[2,8,7],[7,1,3],[1,9,5]], # BB: 0
        [[5,5],[5,5],[5,5]], # BB: 0
        [[10]], # BB: 0
        [[]], # BB: 0
        [[1,2,3],[4,5,6]] # BB: 0
    ];
    test_solutions: list[int] = [6, 10, 17, 10, 10, 0, 15]; # BB: 0

    for i in range(len(test_accounts)) { # BB: 0
        accounts: list[list[int]] = test_accounts[i]; # BB: 1 Execution frequency: 0 Total execution time: 0.000 ms
        solution: int = test_solutions[i]; # BB: 1
        max_wealth: int = 0; # BB: 1
        wealth: int = 0; # BB: 1

        for customer in accounts { # BB: 1
            for money in customer { # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
                wealth += money; # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
            }
            if (wealth > max_wealth) { # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
                max_wealth:int = wealth; # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
            }
        }

        print(max_wealth == solution); # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
    }
}