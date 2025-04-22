import:py random;
#   Instructions: [SETUP_ANNOTATIONS(None)]

with entry {
    threshold:int = 5;
    #   Instructions: [LOAD_CONST(5), STORE_NAME(threshold) [no line], LOAD_CONST(int) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(threshold) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(range) [no line], LOAD_CONST(10) [no line], CALL(1) [no line], GET_ITER(None) [no line]]
    for i in range(10) {
    #   Instructions: [PUSH_NULL(None), END_FOR(None), RETURN_CONST(None) [no line], SWAP(2) [no line], POP_TOP(None) [no line], STORE_FAST(_) [no line], RERAISE(0) [no line], SWAP(2) [no line], POP_TOP(None) [no line], STORE_FAST(n) [no line]]
        nums:list[int] = [random.randint(0, 10) for _ in range(6)];
        #   Instructions: [FOR_ITER(280) [no line], STORE_NAME(i) [no line], PUSH_NULL(None), LOAD_NAME(range) [no line], LOAD_CONST(6) [no line], CALL(1) [no line], GET_ITER(None) [no line], LOAD_FAST_AND_CLEAR(_) [no line], SWAP(2) [no line], BUILD_LIST(0) [no line], SWAP(2) [no line], SWAP(2)]
        above_threshold:list[int] = [n for n in nums if n > threshold];
        #   Instructions: [END_FOR(None) [no line], SWAP(2) [no line], STORE_FAST(_) [no line], STORE_NAME(nums) [no line], LOAD_CONST(list[int]) [no line], LOAD_NAME(__annotations__) [no line], LOAD_CONST(nums) [no line], STORE_SUBSCR(None) [no line], LOAD_NAME(nums), GET_ITER(None) [no line], LOAD_FAST_AND_CLEAR(n) [no line], SWAP(2) [no line], BUILD_LIST(0) [no line], SWAP(2) [no line], SWAP(2)]
    }
}