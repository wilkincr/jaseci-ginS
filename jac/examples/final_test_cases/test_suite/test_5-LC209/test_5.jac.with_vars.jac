with entry {
#   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 10×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 2×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)
    test_targets: list[int] = [7, 15, 4, 8, 11, 3, 100];
    test_nums: list[list[int]] = [
        [2,3,1,2,4,3],
        [1,2,3,4,5],
        [1,4,4],
        [1,4,4],
        [1,2,3,4,5],
        [1,1,1,1],
        [1,1,1,1]
    ];
    test_solutions: list[int] = [2, 2, 1, 2, 3, 3, 0];

    for i in range(len(test_targets)) {
        target: int = test_targets[i];
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 13×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 13×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 13×)  target = 7 (seen 12×)  nums = [2, 3, 1, 2, 4, 3] (seen 11×)  solution = 2 (seen 10×)  n = 6 (seen 9×)  min_length = 7 (seen 6×)  min_length = 4 (seen 2×)  current_sum = 6 (seen 3×)  current_sum = 0 (seen 2×)  current_sum = 2 (seen 1×)  current_sum = 5 (seen 1×)  left = 0 (seen 4×)  left = 1 (seen 1×)  left = 2 (seen 1×)
        nums: list[int] = test_nums[i];
        solution: int = test_solutions[i];
        n: int = len(nums);
        min_length: int = n + 1;
        current_sum: int = 0;
        left: int = 0;

        for right in range(n) {
            current_sum += nums[right];
            #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 14×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 14×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 14×)  target = 7 (seen 14×)  nums = [2, 3, 1, 2, 4, 3] (seen 14×)  solution = 2 (seen 14×)  n = 6 (seen 14×)  min_length = 7 (seen 8×)  min_length = 4 (seen 6×)  current_sum = 6 (seen 6×)  current_sum = 2 (seen 2×)  current_sum = 5 (seen 2×)  current_sum = 0 (seen 1×)  current_sum = 8 (seen 1×)  left = 0 (seen 8×)  left = 1 (seen 3×)  left = 2 (seen 3×)
            while (current_sum >= target) {
                min_length:int = min(min_length, right - left + 1);
                #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 8×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 8×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 8×)  target = 7 (seen 8×)  nums = [2, 3, 1, 2, 4, 3] (seen 8×)  solution = 2 (seen 8×)  n = 6 (seen 8×)  min_length = 4 (seen 7×)  min_length = 7 (seen 1×)  current_sum = 8 (seen 2×)  current_sum = 10 (seen 2×)  current_sum = 9 (seen 2×)  current_sum = 6 (seen 2×)  left = 0 (seen 3×)  left = 1 (seen 3×)  left = 2 (seen 2×)
                current_sum -= nums[right];
                left += 1;
            }
        }

        if (min_length == n + 1) {
            answer: int = 0;
        }
        else {
            answer: int = min_length;
        }
        print(answer == solution);
    }
}