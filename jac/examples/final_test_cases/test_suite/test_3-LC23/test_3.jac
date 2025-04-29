import:py from jaclang.runtimelib.gins.smart_assert, smart_assert;

with entry {
    test_lists: list[list[list[int]]] = [
        [[1,4,5],[1,3,4],[2,6]],
        [],
        [[]],
        [[1,3,5,7],[2,4,6,8]],
        [[1],[0]],
        [[1,2],[1,3,4],[2,6]],
        [[5,6,7],[1,2,3,8]]
    ];
    test_solutions: list[list[int]] = [
        [1,1,2,3,4,4,5,6],
        [],
        [],
        [1,2,3,4,5,6,7,8],
        [0,1],
        [1,1,2,2,3,4,6],
        [1,2,3,5,6,7,8]
    ];

    for i in range(len(test_lists)) {
        lists: list[list[int]] = test_lists[i];
        solution: list[int] = test_solutions[i];

        merged: list[int] = [];

        pointers: list[int] = [];
        for _ in range(len(lists)) {
            pointers.append(0);
        }

        while True {
            min_val: int = 0;
            min_idx: int = -1;

            for j in range(len(lists)) {
                if pointers[j] < len(lists[j]) {
                    if lists[j][pointers[j]] < min_val {
                        min_val:int = lists[j][pointers[j]];
                        min_idx:int = j;
                    }
                }
            }

            if min_idx == -1 {
                break;
            }

            merged.append(min_val);
            pointers[min_idx] += 1;
        }

        smart_assert(merged == solution);
    }
}
