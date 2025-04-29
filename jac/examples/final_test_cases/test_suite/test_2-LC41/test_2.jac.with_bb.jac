with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.005 ms
    test_nums: list[list[int]] = [ # BB: 0
        [1,2,0], # BB: 0
        [3,4,-1,1], # BB: 0
        [7,8,9,11,12], # BB: 0
        [1,1], # BB: 0
        [], # BB: 0
        [2], # BB: 0
        [1,2,3,4,5] # BB: 0
    ];
    test_solutions: list[int] = [3, 2, 1, 2, 1, 1, 6]; # BB: 0

    for i in range(len(test_nums)) { # BB: 0
        nums: list[int] = test_nums[i]; # BB: 1 Execution frequency: 1 Total execution time: 0.000 ms
        solution: int = test_solutions[i]; # BB: 1
        
        n: int = len(nums); # BB: 1

        for j in range(n) { # BB: 1
            while (nums[j] > 0 and nums[j] <= n and nums[nums[j]-1] != nums[j]) { # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
                temp: int = nums[nums[j]-1]; # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
                nums[nums[j]-1]:int = nums[j]; # BB: 8
                nums[j]:int = temp; # BB: 8
            }
        }

        missing: int = n + 1; # BB: 15 Execution frequency: 0 Total execution time: 0.000 ms
        for j in range(n) { # BB: 15
            if (nums[j] != j) { # BB: 16 Execution frequency: 0 Total execution time: 0.000 ms
                missing:int = j; # BB: 18 Execution frequency: 0 Total execution time: 0.000 ms
                break; # BB: 18
            }
        }

        print(missing == solution); # BB: 20 Execution frequency: 0 Total execution time: 0.000 ms
    }
}