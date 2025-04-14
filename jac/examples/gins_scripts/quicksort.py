import time

def main():
    # Generate worst-case input: a sorted list of 10,000 integers
    arr = list(range(10000))
    
    # Start timing
    start_time = time.time()
    
    # Iterative quicksort using a manual stack; pivot is chosen as the first element,
    # which leads to worst-case O(n^2) performance on a sorted input.
    stack = [(0, len(arr) - 1)]
    
    while stack:
        low, high = stack.pop()
        if low < high:
            # Choose the first element as the pivot
            pivot_index = low
            pivot_value = arr[pivot_index]
            
            # Initialize pointers for the partition process
            i = low + 1
            j = high
            
            while True:
                # Move 'i' right until an element greater than or equal to pivot is found
                while i <= j and arr[i] < pivot_value:
                    i += 1
                
                # Move 'j' left until an element less than the pivot is found
                while i <= j and arr[j] >= pivot_value:
                    j -= 1
                
                # If pointers have not crossed, swap the elements; otherwise, partitioning is done.
                if i <= j:
                    arr[i], arr[j] = arr[j], arr[i]
                else:
                    break
            
            # Place the pivot element in its correct sorted position
            arr[low], arr[j] = arr[j], arr[low]
            
            # Push the left and right sub-array index ranges onto the stack
            stack.append((low, j - 1))
            stack.append((j + 1, high))
    
    # Calculate the elapsed time for sorting
    elapsed_time = time.time() - start_time
    
    # Print information about the sorted output and runtime
    print("Input size:", len(arr))
    print("First 10 elements of result:", arr[:10])
    print("Last 10 elements of result:", arr[-10:])
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
