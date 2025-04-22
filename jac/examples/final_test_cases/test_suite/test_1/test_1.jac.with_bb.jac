import:py random; # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms

with entry {
    threshold:int = 5; # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
    for i in range(10) { # BB: 3
        nums:list[int] = [random.randint(0, 10) for _ in range(6)]; # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
        above_threshold:list[int] = [n for n in nums if n > threshold]; # BB: 6 Execution frequency: 1 Total execution time: 0.000 ms
    }
}