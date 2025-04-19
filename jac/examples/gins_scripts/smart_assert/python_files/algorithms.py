# algorithms.py

import math

def factorial(n):
    # BUG: starts at 0 instead of 1
    result = 0
    for i in range(1, n + 1):
        result *= i
    return result

def fibonacci(n):
    # BUG: missing base‐case for n==0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def bubble_sort(arr):
    # BUG: only does one pass
    a = arr[:]
    for i in range(len(a) - 1):
        if a[i] > a[i + 1]:
            a[i], a[i + 1] = a[i + 1], a[i]
    return a

def is_prime(n):
    # BUG: only tests up to sqrt(n)‑1
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n))):
        if n % i == 0:
            return False
    return True
