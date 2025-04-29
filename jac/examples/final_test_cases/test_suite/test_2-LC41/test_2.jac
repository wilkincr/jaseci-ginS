import:py from jaclang.runtimelib.gins.smart_assert, smart_assert;

with entry {
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
        solution: int = test_solutions[i];
        
        n: int = len(nums);

        for j in range(n) {
            while (nums[j] > 0 and nums[j] <= n and nums[nums[j]-1] != nums[j]) {
                temp: int = nums[nums[j]-1];
                nums[nums[j]-1]:int = nums[j];
                nums[j]:int = temp;
            }
        }

        missing: int = n + 1;
        for j in range(n) {
            if (nums[j] != j) {
                missing:int = j;
                break;
            }
        }

        smart_assert(missing == solution);
    }
}