import:py random;

with entry {
    threshold:int = 5;
    #   Variable values in this block:   threshold = 5 (seen 1×)
    for i in range(10) {
        nums:list[int] = [random.randint(0, 10) for _ in range(6)];
        #   Variable values in this block:   threshold = 5 (seen 1×)
        above_threshold:list[int] = [n for n in nums if n > threshold];
        #   Variable values in this block:   threshold = 5 (seen 1×)
    }
}