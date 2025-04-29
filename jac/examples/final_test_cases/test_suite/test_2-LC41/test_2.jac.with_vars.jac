with entry {
#   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 3×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 2×)  nums = [1, 2, 0] (seen 1×)  solution = 3 (seen 1×)  n = 3 (seen 1×)  missing = 0 (seen 1×)
    test_nums: list[list[int]] = [
        [1,2,0],
        [3,4,-1,1],
        [7,8,9,11,12],
        [1,1],
        [],
        [2],
        [1,2,3,4,5]
    ];
    test_solutions: list[int] = [3, 2, 1, 2, 1, 1, 6];

    for i in range(len(test_nums)) {
        nums: list[int] = test_nums[i];
        #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 11×)  test_nums = [[1, 2, 0], [-1, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 12×)  nums = [1, 2, 0] (seen 7×)  nums = [3, 4, -1, 1] (seen 3×)  nums = [-1, 4, 3, 1] (seen 1×)  solution = 3 (seen 7×)  solution = 2 (seen 3×)  n = 3 (seen 7×)  n = 4 (seen 2×)  missing = 0 (seen 5×)  temp = -1 (seen 1×)
        solution: int = test_solutions[i];
        
        n: int = len(nums);

        for j in range(n) {
            while (nums[j] > 0 and nums[j] <= n and nums[nums[j]-1] != nums[j]) {
            #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 4×)  test_nums = [[1, 2, 0], [-1, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 6×)  nums = [1, 2, 0] (seen 3×)  nums = [-1, 4, 3, 1] (seen 2×)  nums = [3, 4, -1, 1] (seen 1×)  solution = 3 (seen 3×)  solution = 2 (seen 3×)  n = 3 (seen 3×)  n = 4 (seen 3×)  missing = 0 (seen 3×)  temp = -1 (seen 2×)
                temp: int = nums[nums[j]-1];
                #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_nums = [[1, 2, 0], [3, 4, 3, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 3×)  nums = [3, 4, -1, 1] (seen 2×)  nums = [3, 4, 3, 1] (seen 1×)  solution = 2 (seen 3×)  n = 4 (seen 3×)  missing = 0 (seen 3×)  temp = -1 (seen 2×)
                nums[nums[j]-1]:int = nums[j];
                nums[j]:int = temp;
            }
        }

        missing: int = n + 1;
        #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 2×)  nums = [1, 2, 0] (seen 2×)  solution = 3 (seen 2×)  n = 3 (seen 2×)  missing = 4 (seen 1×)
        for j in range(n) {
            if (nums[j] != j) {
            #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 1×)  nums = [1, 2, 0] (seen 1×)  solution = 3 (seen 1×)  n = 3 (seen 1×)  missing = 4 (seen 1×)
                missing:int = j;
                #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 2×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 2×)  nums = [1, 2, 0] (seen 2×)  solution = 3 (seen 2×)  n = 3 (seen 2×)  missing = 4 (seen 1×)  missing = 0 (seen 1×)
                break;
            }
        }

        print(missing == solution);
        #   Variable values in this block:   test_nums = [[1, 2, 0], [3, 4, -1, 1], [7, 8, 9, 11, 12], [1, 1], [], [2], [1, 2, 3, 4, 5]] (seen 1×)  test_solutions = [3, 2, 1, 2, 1, 1, 6] (seen 1×)  nums = [1, 2, 0] (seen 1×)  solution = 3 (seen 1×)  n = 3 (seen 1×)  missing = 0 (seen 1×)
    }
}