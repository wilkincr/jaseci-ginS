with entry {
    test_nums: list[list[int]] = [
        [2,7,9,3,1],
        [1,2,3,1],
        [2,1,1,2],
        [5],
        [],
        [2,1,1,2,1],
        [10,1,1,10]
    ];
    test_solutions: list[int] = [12, 4, 4, 5, 0, 5, 20];

    for i in range(len(test_nums)) {
        nums: list[int] = test_nums[i];
        solution: int = test_solutions[i];
        answer: int = 0;

        if (len(nums) == 0) {
            answer: int = 0;
        }
        elif (len(nums) == 1) {
            answer:int = nums[0];
        }
        else {
            n: int = len(nums);
            dp: list[int] = [0]*n;
            dp[0]:int = nums[0];
            dp[1]:int = max(nums[0], nums[1]);
            for j in range(2, n) {
                dp[j]:int = max(dp[j-1] + nums[j], dp[j-2] + nums[j]);
            }
            answer:int = dp[-1];
        }

        print(answer == solution);
    }
}