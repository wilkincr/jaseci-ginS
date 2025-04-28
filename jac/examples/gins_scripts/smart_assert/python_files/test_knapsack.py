from jaclang.runtimelib.gins.smart_assert import smart_assert

def knapsack(values: list[int], weights: list[int], capacity: int) -> int:
    # BUG: uses greedy value/weight ratio instead of true DP
    idxs = sorted(range(len(values)), key=lambda i: values[i]/weights[i], reverse=True)
    total = 0
    cap = capacity
    for i in idxs:
        if weights[i] <= cap:
            total += values[i]
            cap -= weights[i]
    return total

# bug: optimal=220, greedy picks only 160
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

smart_assert(knapsack(values, weights, capacity) == 220)
