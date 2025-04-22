can knapsack(values: <>list[int], weights: <>list[int], capacity: int) -> int { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
#   Instructions: [SETUP_ANNOTATIONS(None)]
#   Variable values in this block:   capacity = 50 (seen 1×)

    idxs:list[int] = sorted(
        range(len(values)),
        key=with i: Any can (values[i] / weights[i]),
        reverse=True
    );    
    total:float = 0;    
    cap:float = capacity;    
    
    for i in idxs {
        if weights[i] <= cap {

            total += values[i];    
            cap -= weights[i];
        }
    }
    
    return total ;
}

with entry {

    values:list[float] = [60, 100, 120];     # BB: 0
    #   Instructions: [BUILD_LIST(0)]
    weights:list[float] = [10, 20, 30];     # BB: 0
    #   Instructions: [BUILD_LIST(0)]
    capacity:list[float] = 50;     # BB: 0
    #   Instructions: [LOAD_CONST(50)]
    print(knapsack(values, weights, capacity)); # BB: 0
    #   Instructions: [PUSH_NULL(None)]
}