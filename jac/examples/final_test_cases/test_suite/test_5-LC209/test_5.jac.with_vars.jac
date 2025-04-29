with entry {
#   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 11×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 3×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 2×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)  answer = 4 (seen 1×)
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
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 24×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 24×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 24×)  target = 7 (seen 14×)  target = 15 (seen 9×)  nums = [2, 3, 1, 2, 4, 3] (seen 14×)  nums = [1, 2, 3, 4, 5] (seen 8×)  solution = 2 (seen 21×)  n = 6 (seen 14×)  n = 5 (seen 6×)  min_length = 4 (seen 8×)  min_length = 7 (seen 6×)  min_length = 6 (seen 5×)  current_sum = 6 (seen 10×)  current_sum = 0 (seen 4×)  current_sum = 2 (seen 1×)  current_sum = 5 (seen 1×)  current_sum = 1 (seen 1×)  left = 3 (seen 8×)  left = 0 (seen 7×)  left = 1 (seen 1×)  left = 2 (seen 1×)  answer = 4 (seen 10×)
        nums: list[int] = test_nums[i];
        solution: int = test_solutions[i];
        n: int = len(nums);
        min_length: int = n + 1;
        current_sum: int = 0;
        left: int = 0;

        for right in range(n) {
            current_sum += nums[right];
            #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 21×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 21×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 21×)  target = 7 (seen 15×)  target = 15 (seen 6×)  nums = [2, 3, 1, 2, 4, 3] (seen 15×)  nums = [1, 2, 3, 4, 5] (seen 6×)  solution = 2 (seen 21×)  n = 6 (seen 15×)  n = 5 (seen 6×)  min_length = 7 (seen 8×)  min_length = 4 (seen 7×)  min_length = 6 (seen 6×)  current_sum = 6 (seen 8×)  current_sum = 0 (seen 2×)  current_sum = 2 (seen 2×)  current_sum = 5 (seen 2×)  current_sum = 1 (seen 2×)  left = 0 (seen 14×)  left = 1 (seen 3×)  left = 2 (seen 3×)  left = 3 (seen 1×)  answer = 4 (seen 6×)
            while (current_sum >= target) {
                min_length:int = min(min_length, right - left + 1);
                #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 9×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 9×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 9×)  target = 7 (seen 9×)  nums = [2, 3, 1, 2, 4, 3] (seen 9×)  solution = 2 (seen 9×)  n = 6 (seen 9×)  min_length = 4 (seen 8×)  min_length = 7 (seen 1×)  current_sum = 6 (seen 3×)  current_sum = 8 (seen 2×)  current_sum = 10 (seen 2×)  current_sum = 9 (seen 2×)  left = 0 (seen 3×)  left = 1 (seen 3×)  left = 2 (seen 3×)
                current_sum -= nums[right];
                left += 1;
            }
        }

        if (min_length == n + 1) {
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 1×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 1×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)
            answer: int = 0;
        }
        else {
            answer: int = min_length;
            #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 1×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 1×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)
        }
        print(answer == solution);
        #   Variable values in this block:   test_targets = [7, 15, 4, 8, 11, 3, 100] (seen 1×)  test_nums = [[2, 3, 1, 2, 4, 3], [1, 2, 3, 4, 5], [1, 4, 4], [1, 4, 4], [1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 1, 1, 1]] (seen 1×)  test_solutions = [2, 2, 1, 2, 3, 3, 0] (seen 1×)  target = 7 (seen 1×)  nums = [2, 3, 1, 2, 4, 3] (seen 1×)  solution = 2 (seen 1×)  n = 6 (seen 1×)  min_length = 4 (seen 1×)  current_sum = 6 (seen 1×)  left = 3 (seen 1×)  answer = 4 (seen 1×)
    }
}