with entry {
#   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 11×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 3×)  nums = [1, 2, 3, 4] (seen 1×)  nums = [1, 1, 1, 1, 1] (seen 1×)  solution = [1, 3, 6, 10] (seen 1×)  solution = [1, 2, 3, 4, 5] (seen 1×)  running_sum = [1, 3, 6, 10] (seen 1×)  running_sum = [1, 2, 3, 4, 5] (seen 1×)  total = 10 (seen 1×)  total = 5 (seen 1×)
    test_nums: list[list[int]] = [
        [1,2,3,4],
        [1,1,1,1,1],
        [3,1,2,10,1],
        [0],
        [],
        [5,10,15],
        [1,2]
    ];
    test_solutions: list[list[int]] = [
        [1,3,6,10],
        [1,2,3,4,5],
        [3,4,6,16,17],
        [0],
        [],
        [5,15,30],
        [1,3]
    ];

    for i in range(len(test_nums)) {
        nums: list[int] = test_nums[i];
        #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 25×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 25×)  nums = [1, 1, 1, 1, 1] (seen 10×)  nums = [1, 2, 3, 4] (seen 9×)  nums = [3, 1, 2, 10, 1] (seen 5×)  solution = [1, 2, 3, 4, 5] (seen 10×)  solution = [1, 3, 6, 10] (seen 9×)  solution = [3, 4, 6, 16, 17] (seen 4×)  running_sum = [] (seen 6×)  running_sum = [1, 3, 6, 10] (seen 4×)  running_sum = [1, 2, 3, 4, 5] (seen 4×)  running_sum = [1] (seen 2×)  running_sum = [1, 3] (seen 1×)  total = 10 (seen 5×)  total = 5 (seen 5×)  total = 0 (seen 3×)  total = 3 (seen 3×)  total = 1 (seen 2×)
        solution: list[int] = test_solutions[i];
        running_sum: list[int] = [];
        total: int = 0;

        for elt in nums {
            total += elt;
            #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 21×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 21×)  nums = [1, 1, 1, 1, 1] (seen 10×)  nums = [1, 2, 3, 4] (seen 8×)  nums = [3, 1, 2, 10, 1] (seen 3×)  solution = [1, 2, 3, 4, 5] (seen 10×)  solution = [1, 3, 6, 10] (seen 8×)  solution = [3, 4, 6, 16, 17] (seen 3×)  running_sum = [] (seen 6×)  running_sum = [1] (seen 4×)  running_sum = [1, 3] (seen 2×)  running_sum = [1, 3, 6] (seen 2×)  running_sum = [1, 2] (seen 2×)  total = 3 (seen 6×)  total = 1 (seen 4×)  total = 0 (seen 3×)  total = 6 (seen 2×)  total = 2 (seen 2×)
            running_sum.append(total);
        }

        print(running_sum == solution);
        #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 2×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 2×)  nums = [1, 2, 3, 4] (seen 1×)  nums = [1, 1, 1, 1, 1] (seen 1×)  solution = [1, 3, 6, 10] (seen 1×)  solution = [1, 2, 3, 4, 5] (seen 1×)  running_sum = [1, 3, 6, 10] (seen 1×)  running_sum = [1, 2, 3, 4, 5] (seen 1×)  total = 10 (seen 1×)  total = 5 (seen 1×)
    }
}