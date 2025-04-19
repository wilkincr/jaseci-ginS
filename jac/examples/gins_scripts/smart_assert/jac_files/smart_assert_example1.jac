import:py random;
import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

with entry {
    threshold:int = 5;
    for i in range(10) {
        nums = [random.randint(0, 10) for _ in range(6)];
        above_threshold = [n for n in nums if n > threshold];
        smart_assert(len(above_threshold) >= len(nums) // 2);
    }
}