with entry { # BB: 0 Transitions: BB0->BB1
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
        lists: list[list[int]] = test_lists[i]; # BB: 1 Transitions: BB1->BB14, BB1->BB2
        solution: list[int] = test_solutions[i]; # BB: 1

        merged: list[int] = []; # BB: 1

        pointers: list[int] = []; # BB: 1
        for _ in range(len(lists)) { # BB: 1
            pointers.append(0); # BB: 2 Transitions: BB2->BB3, BB2->BB2
        }

        while True { # BB: 3 Transitions: BB3->BB4
            min_val: int = 2147483647; # BB: 4 Transitions: BB4->BB5
            min_idx: int = -1; # BB: 4

            for j in range(len(lists)) { # BB: 4
                if pointers[j] < len(lists[j]) { # BB: 5 Transitions: BB5->BB10, BB5->BB7, BB5->BB6
                    if lists[j][pointers[j]] < min_val { # BB: 7 Transitions: BB7->BB9, BB7->BB8
                        min_val:int = lists[j][pointers[j]]; # BB: 9 Transitions: BB9->BB5
                        min_idx:int = j; # BB: 9
                    }
                }
            }

            if min_idx == -1 { # BB: 10 Transitions: BB10->BB12, BB10->BB11
                break; # BB: 11 Transitions: BB11->BB13
            }

            merged.append(min_val); # BB: 12 Transitions: BB12->BB4
            pointers[min_idx] += 1; # BB: 12
        }

        print(merged == solution); # BB: 13 Transitions: BB13->BB1
    }
}