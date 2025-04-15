
    # A simple counter function
    def count_to(n: int) -> list[int]:
        Return a list of numbers from 1 to n.
        return list(range(1, n+1))
    
    # Create a global variable with the result
    glob result = count_to(10)
    print(f"Counted to 10: {result}")
    