with entry { # BB: 0 Execution frequency: 0 Total execution time: 0.000 ms
    test_nums: list[list[int]] = [ # BB: 0
        [2,7,9,3,1], # BB: 0
        [1,2,3,1], # BB: 0
        [2,1,1,2], # BB: 0
        [5], # BB: 0
        [], # BB: 0
        [2,1,1,2,1], # BB: 0
        [10,1,1,10] # BB: 0
    ];
    test_solutions: list[int] = [12, 4, 4, 5, 0, 5, 20]; # BB: 0

    for i in range(len(test_nums)) { # BB: 0
        nums: list[int] = test_nums[i]; # BB: 1 Execution frequency: 0 Total execution time: 0.000 ms
        solution: int = test_solutions[i]; # BB: 1
        answer: int = 0; # BB: 1

        if (len(nums) == 0) { # BB: 1
            answer: int = 0; # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
        }
        elif (len(nums) == 1) { # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
            answer:int = nums[0]; # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
        }
        else {
            n: int = len(nums); # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
            dp: list[int] = [0]*n; # BB: 5
            dp[0]:int = nums[0]; # BB: 5
            dp[1]:int = max(nums[0], nums[1]); # BB: 5
            for j in range(2, n) { # BB: 5
                dp[j]:int = max(dp[j-1] + nums[j], dp[j-2] + nums[j]); # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
            }
            answer:int = dp[-1]; # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
        }

        print(answer == solution); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
    }
}