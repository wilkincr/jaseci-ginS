with entry { # BB: 0 Transitions: BB0->BB1
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
        nums: list[int] = test_nums[i]; # BB: 1 Transitions: BB1->BB21, BB1->BB2
        solution: int = test_solutions[i]; # BB: 1
        
        n: int = len(nums); # BB: 1

        for j in range(n) { # BB: 1
            while (nums[j] > 0 and nums[j] <= n and nums[nums[j]-1] != nums[j]) { # BB: 2 Transitions: BB2->BB15, BB2->BB4, BB2->BB3
                temp: int = nums[nums[j]-1]; # BB: 8 Transitions: BB8->BB10, BB8->BB9
                nums[nums[j]-1]:int = nums[j]; # BB: 8
                nums[j]:int = temp; # BB: 8
            }
        }

        missing: int = n + 1; # BB: 15 Transitions: BB15->BB16
        for j in range(n) { # BB: 15
            if (nums[j] != j) { # BB: 16 Transitions: BB16->BB19, BB16->BB18, BB16->BB17
                missing:int = j; # BB: 18 Transitions: BB18->BB20
                break; # BB: 18
            }
        }

        print(missing == solution); # BB: 20 Transitions: BB20->BB1
    }
}