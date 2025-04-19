from jaclang.runtimelib.gins.smart_assert import smart_assert

def power(x: float, n: int) -> float:
    # BUG: defines 0-th power as 0 instead of 1
    if n == 0:
        return 0.0
    result = 1.0
    for _ in range(n):
        result *= x
    return result

testcases = [
    (2.0, 3, 8.0),
    (5.0, 2, 25.0),
    (10.0, 0, 1.0),  #this test case will fail
    (3.0, 1, 3.0),
]

for base, exp, expected in testcases:
    got = power(base, exp)
    smart_assert(got == expected)

