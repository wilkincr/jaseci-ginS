import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

can knapsack(values: <>list[int], weights: <>list[int], capacity: int) -> int {

    idxs = sorted(
        range(len(values)),
        key=with i: Any can (values[i] / weights[i]),
        reverse=True
    );    
    total = 0;    
    cap = capacity;    
    
    for i in idxs {
        if weights[i] <= cap {

            total += values[i];    
            cap -= weights[i];
        }
    }
    
    return total ;
}

with entry {

    values = [60, 100, 120];    
    weights = [10, 20, 30];    
    capacity = 50;    
    smart_assert(knapsack(values, weights, capacity) == 220) ;
}

