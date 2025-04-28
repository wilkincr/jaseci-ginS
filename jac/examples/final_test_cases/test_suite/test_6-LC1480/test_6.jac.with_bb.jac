with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.004 ms
    test_nums: list[list[int]] = [ # BB: 0
        [1,2,3,4], # BB: 0
        [1,1,1,1,1], # BB: 0
        [3,1,2,10,1], # BB: 0
        [0], # BB: 0
        [], # BB: 0
        [5,10,15], # BB: 0
        [1,2] # BB: 0
    ];
    test_solutions: list[list[int]] = [ # BB: 0
        [1,3,6,10], # BB: 0
        [1,2,3,4,5], # BB: 0
        [3,4,6,16,17], # BB: 0
        [0], # BB: 0
        [], # BB: 0
        [5,15,30], # BB: 0
        [1,3] # BB: 0
    ];

    for i in range(len(test_nums)) { # BB: 0
        nums: list[int] = test_nums[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.003 ms
        solution: list[int] = test_solutions[i]; # BB: 1
        running_sum: list[int] = []; # BB: 1
        total: int = 0; # BB: 1

        for elt in nums { # BB: 1
            total += elt; # BB: 2 Execution frequency: 1 Total execution time: 0.000 ms
            running_sum.append(total); # BB: 2
        }

        print(running_sum == solution); # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
    }
}