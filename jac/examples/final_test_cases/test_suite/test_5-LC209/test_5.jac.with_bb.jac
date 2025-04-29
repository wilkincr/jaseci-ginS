with entry { # BB: 0 Transitions: BB0->BB1
    test_targets: list[int] = [7, 15, 4, 8, 11, 3, 100]; # BB: 0
    test_nums: list[list[int]] = [ # BB: 0
        [2,3,1,2,4,3], # BB: 0
        [1,2,3,4,5], # BB: 0
        [1,4,4], # BB: 0
        [1,4,4], # BB: 0
        [1,2,3,4,5], # BB: 0
        [1,1,1,1], # BB: 0
        [1,1,1,1] # BB: 0
    ];
    test_solutions: list[int] = [2, 2, 1, 2, 3, 3, 0]; # BB: 0

    for i in range(len(test_targets)) { # BB: 0
        target: int = test_targets[i]; # BB: 1 Transitions: BB1->BB11, BB1->BB2
        nums: list[int] = test_nums[i]; # BB: 1
        solution: int = test_solutions[i]; # BB: 1
        n: int = len(nums); # BB: 1
        min_length: int = n + 1; # BB: 1
        current_sum: int = 0; # BB: 1
        left: int = 0; # BB: 1

        for right in range(n) { # BB: 1
            current_sum += nums[right]; # BB: 2 Transitions: BB2->BB7, BB2->BB4, BB2->BB3
            while (current_sum >= target) { # BB: 2
                min_length:int = min(min_length, right - left + 1); # BB: 4 Transitions: BB4->BB6, BB4->BB5
                current_sum -= nums[right]; # BB: 4
                left += 1; # BB: 4
            }
        }

        if (min_length == n + 1) { # BB: 7 Transitions: BB7->BB9, BB7->BB8
            answer: int = 0; # BB: 8 Transitions: BB8->BB10
        }
        else {
            answer: int = min_length; # BB: 9 Transitions: BB9->BB10
        }
        print(answer == solution); # BB: 10 Transitions: BB10->BB1
    }
}