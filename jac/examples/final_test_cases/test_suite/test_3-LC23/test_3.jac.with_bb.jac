with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
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

        pointers: list[int] = []; # BB: 1
        for _ in range(len(lists)) { # BB: 1
            pointers.append(0); # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
        }

        while True { # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
            min_val: int = 2147483647; # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
            min_idx: int = -1; # BB: 4

            for j in range(len(lists)) { # BB: 4
                if pointers[j] < len(lists[j]) { # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
                    if lists[j][pointers[j]] < min_val { # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
                        min_val:int = lists[j][pointers[j]]; # BB: 9 Execution frequency: 0 Total execution time: 0.000 ms
                        min_idx:int = j; # BB: 9
                    }
                }
            }

            if min_idx == -1 { # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
                break; # BB: 11 Execution frequency: 0 Total execution time: 0.000 ms
            }

            merged.append(min_val); # BB: 12 Execution frequency: 0 Total execution time: 0.000 ms
            pointers[min_idx] += 1; # BB: 12
        }

        print(merged == solution); # BB: 13 Execution frequency: 0 Total execution time: 0.000 ms
    }
}