with entry {
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
        solution: list[int] = test_solutions[i];
        running_sum: list[int] = [];
        total: int = 0;

        for elt in nums {
            running_sum.append(total);
            total += elt;
        }

        print(running_sum == solution);
    }
}
