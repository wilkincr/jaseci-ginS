from jaclang.runtimelib.gins.smart_assert import smart_assert
import math

def is_prime(n):
    # BUG: only tests up to sqrt(n)‑1
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n))):
        if n % i == 0:
            return False
    return True

testcases = [
    (0, False),
    (1, False),
    (4, False),
    (9, False),
    (17, True),
    (18, False),
    (19, True),
]

for idx, (inp, expected) in enumerate(testcases, 1):
    got = is_prime(inp)
    smart_assert(got == expected)
