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
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 12×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 12×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 12×)  target = 7 (seen 11×)  nums = [2, 3, 1, 2, 4, 3] (seen 10×)  solution = 2 (seen 9×)  n = 6 (seen 8×)  min_length = 7 (seen 6×)  min_length = 4 (seen 1×)  current_sum = 0 (seen 2×)  current_sum = 6 (seen 2×)  current_sum = 2 (seen 1×)  current_sum = 5 (seen 1×)  left = 0 (seen 4×)  left = 1 (seen 1×)
        nums: list[int] = test_nums[i];
        solution: int = test_solutions[i];
        n: int = len(nums);
        min_length: int = n + 1;
        current_sum: int = 0;
        left: int = 0;

        for right in range(n) {
            current_sum += nums[right];
            #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 9×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 9×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 9×)  target = 7 (seen 9×)  nums = [2, 3, 1, 2, 4, 3] (seen 9×)  solution = 2 (seen 9×)  n = 6 (seen 9×)  min_length = 7 (seen 8×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 3×)  current_sum = 2 (seen 2×)  current_sum = 5 (seen 2×)  current_sum = 0 (seen 1×)  current_sum = 8 (seen 1×)  left = 0 (seen 8×)  left = 1 (seen 1×)
            while (current_sum >= target) {
                min_length:int = min(min_length, right - left + 1);
                #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 3×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 3×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 3×)  target = 7 (seen 3×)  nums = [2, 3, 1, 2, 4, 3] (seen 3×)  solution = 2 (seen 3×)  n = 6 (seen 3×)  min_length = 4 (seen 2×)  min_length = 7 (seen 1×)  current_sum = 8 (seen 2×)  current_sum = 6 (seen 1×)  left = 0 (seen 3×)
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