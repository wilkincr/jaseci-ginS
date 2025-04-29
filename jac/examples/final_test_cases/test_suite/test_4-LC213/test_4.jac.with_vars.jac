with entry {
#   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 2×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 1×)
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
        #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 4×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 4×)  nums = [2, 7, 9, 3, 1] (seen 3×)  solution = 12 (seen 2×)  answer = 0 (seen 1×)
        solution: int = test_solutions[i];
        answer: int = 0;

        if (len(nums) == 0) {
            answer: int = 0;
        }
        elif (len(nums) == 1) {
        #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 1×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 1×)  nums = [2, 7, 9, 3, 1] (seen 1×)  solution = 12 (seen 1×)  answer = 0 (seen 1×)
            answer:int = nums[0];
        }
        else {
            n: int = len(nums);
            #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 7×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 7×)  nums = [2, 7, 9, 3, 1] (seen 7×)  solution = 12 (seen 7×)  answer = 0 (seen 7×)  n = 5 (seen 6×)  dp = [0, 0, 0, 0, 0] (seen 1×)  dp = [2, 0, 0, 0, 0] (seen 1×)  dp = [2, 7, 0, 0, 0] (seen 1×)  dp = [2, 7, 16, 0, 0] (seen 1×)  dp = [2, 7, 16, 19, 0] (seen 1×)
            dp: list[int] = [0]*n;
            dp[0]:int = nums[0];
            dp[1]:int = max(nums[0], nums[1]);
            for j in range(2, n) {
                dp[j]:int = max(dp[j-1] + nums[j], dp[j-2] + nums[j]);
                #   Variable values in this block:   test_nums = [[2, 7, 9, 3, 1], [1, 2, 3, 1], [2, 1, 1, 2], [5], [], [2, 1, 1, 2, 1], [10, 1, 1, 10]] (seen 3×)  test_solutions = [12, 4, 4, 5, 0, 5, 20] (seen 3×)  nums = [2, 7, 9, 3, 1] (seen 3×)  solution = 12 (seen 3×)  answer = 0 (seen 3×)  n = 5 (seen 3×)  dp = [2, 7, 0, 0, 0] (seen 1×)  dp = [2, 7, 16, 0, 0] (seen 1×)  dp = [2, 7, 16, 19, 0] (seen 1×)
            }
            answer:int = dp[-1];
        }

        print(answer == solution);
    }
}