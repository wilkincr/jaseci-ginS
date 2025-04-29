with entry {
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
        nums: list[int] = test_nums[i];
        solution: int = test_solutions[i];
        n: int = len(nums);
        min_length: int = n + 1;
        current_sum: int = 0;
        left: int = 0;

        for right in range(n) {
            current_sum += nums[right];
            while (current_sum >= target) {
                min_length:int = min(min_length, right - left + 1);
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
