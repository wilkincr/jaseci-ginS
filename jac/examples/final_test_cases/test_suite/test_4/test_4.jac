can knapsack(values: <>list[int], weights: <>list[int], capacity: int) -> int {

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

    values:list[float] = [60, 100, 120];    
    weights:list[float] = [10, 20, 30];    
    capacity:list[float] = 50;    
    print(knapsack(values, weights, capacity));
}
