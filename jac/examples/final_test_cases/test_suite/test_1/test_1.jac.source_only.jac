import:py random;

with entry {
    threshold:int = 5;
    for i in range(10) {
        nums:list[int] = [random.randint(0, 10) for _ in range(6)];
        above_threshold:list[int] = [n for n in nums if n > threshold];
    }
}