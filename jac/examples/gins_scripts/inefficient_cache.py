import time
import random

def main():
    cache = {}
    
    inputs = []
    for _ in range(5000):
        inputs.append(random.randint(1, 10**6))
    
    start_time = time.time()
    
    results = []
    for num in inputs:
        if num in cache:
            result = cache[num]
        else:
            # Simulate a slow computation
            time.sleep(0.01)
            result = num * num
            cache[num] = result
        results.append(result)
    
    print(f"Computed {len(results)} results.")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()