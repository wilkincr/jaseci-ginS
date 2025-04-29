with entry { # BB: 0 Execution frequency: 0 Total execution time: 0.000 ms
    test_lists: list[list[list[int]]] = [ # BB: 0
        [[1,4,5],[1,3,4],[2,6]], # BB: 0
        [], # BB: 0
        [[]], # BB: 0
        [[1,3,5,7],[2,4,6,8]], # BB: 0
        [[1],[0]], # BB: 0
        [[1,2],[1,3,4],[2,6]], # BB: 0
        [[5,6,7],[1,2,3,8]] # BB: 0
    ];
    test_solutions: list[list[int]] = [ # BB: 0
        [1,1,2,3,4,4,5,6], # BB: 0
        [], # BB: 0
        [], # BB: 0
        [1,2,3,4,5,6,7,8], # BB: 0
        [0,1], # BB: 0
        [1,1,2,2,3,4,6], # BB: 0
        [1,2,3,5,6,7,8] # BB: 0
    ];

    for i in range(len(test_lists)) { # BB: 0
        lists: list[list[int]] = test_lists[i]; # BB: 1 Execution frequency: 0 Total execution time: 0.000 ms
        solution: list[int] = test_solutions[i]; # BB: 1
        merged: list[int] = []; # BB: 1

        for lst in lists { # BB: 1
            merged: list[int] = merged + lst; # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
        }

        merged.sort(); # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
        print(merged == solution); # BB: 3
    }
}