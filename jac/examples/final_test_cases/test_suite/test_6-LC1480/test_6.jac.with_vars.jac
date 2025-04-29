with entry {
#   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 10×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 2×)  nums = [1, 2, 3, 4] (seen 1×)  solution = [1, 3, 6, 10] (seen 1×)  running_sum = [0, 1, 3, 6] (seen 1×)  total = 10 (seen 1×)
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
        #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 17×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 17×)  nums = [1, 2, 3, 4] (seen 9×)  nums = [1, 1, 1, 1, 1] (seen 7×)  solution = [1, 3, 6, 10] (seen 9×)  solution = [1, 2, 3, 4, 5] (seen 6×)  running_sum = [0, 1, 3, 6] (seen 4×)  running_sum = [] (seen 4×)  running_sum = [0] (seen 2×)  running_sum = [0, 1] (seen 2×)  running_sum = [0, 1, 3] (seen 1×)  total = 10 (seen 5×)  total = 0 (seen 2×)  total = 1 (seen 2×)  total = 3 (seen 2×)  total = 6 (seen 1×)
        solution: list[int] = test_solutions[i];
        running_sum: list[int] = [];
        total: int = 0;

        for elt in nums {
            running_sum.append(total);
            #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 16×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 16×)  nums = [1, 2, 3, 4] (seen 8×)  nums = [1, 1, 1, 1, 1] (seen 8×)  solution = [1, 3, 6, 10] (seen 8×)  solution = [1, 2, 3, 4, 5] (seen 8×)  running_sum = [0] (seen 4×)  running_sum = [0, 1] (seen 4×)  running_sum = [] (seen 2×)  running_sum = [0, 1, 3] (seen 2×)  running_sum = [0, 1, 2] (seen 2×)  total = 0 (seen 4×)  total = 1 (seen 4×)  total = 3 (seen 4×)  total = 6 (seen 2×)  total = 2 (seen 2×)
            total += elt;
        }

        print(running_sum == solution);
        #   Variable values in this block:   test_nums = [[1, 2, 3, 4], [1, 1, 1, 1, 1], [3, 1, 2, 10, 1], [0], [], [5, 10, 15], [1, 2]] (seen 1×)  test_solutions = [[1, 3, 6, 10], [1, 2, 3, 4, 5], [3, 4, 6, 16, 17], [0], [], [5, 15, 30], [1, 3]] (seen 1×)  nums = [1, 2, 3, 4] (seen 1×)  solution = [1, 3, 6, 10] (seen 1×)  running_sum = [0, 1, 3, 6] (seen 1×)  total = 10 (seen 1×)
    }
}