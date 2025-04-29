with entry { # BB: 0 Transitions: BB0->BB1
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
        nums: list[int] = test_nums[i]; # BB: 1 Transitions: BB1->BB9, BB1->BB3, BB1->BB2
        solution: int = test_solutions[i]; # BB: 1
        answer: int = 0; # BB: 1

        if (len(nums) == 0) { # BB: 1
            answer: int = 0; # BB: 2 Transitions: BB2->BB8
        }
        elif (len(nums) == 1) { # BB: 3 Transitions: BB3->BB5, BB3->BB4
            answer:int = nums[0]; # BB: 4 Transitions: BB4->BB8
        }
        else {
            n: int = len(nums); # BB: 5 Transitions: BB5->BB6
            dp: list[int] = [0]*n; # BB: 5
            dp[0]:int = nums[0]; # BB: 5
            dp[1]:int = max(nums[0], nums[1]); # BB: 5
            for j in range(2, n) { # BB: 5
                dp[j]:int = max(dp[j-1] + nums[j], dp[j-2] + nums[j]); # BB: 6 Transitions: BB6->BB7, BB6->BB6
            }
            answer:int = dp[-1]; # BB: 7 Transitions: BB7->BB8
        }

        print(answer == solution); # BB: 8 Transitions: BB8->BB1
    }
}