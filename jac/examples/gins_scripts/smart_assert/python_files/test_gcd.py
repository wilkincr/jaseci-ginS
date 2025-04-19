from jaclang.runtimelib.gins.smart_assert import smart_assert
    # BUG: when b==0, returns 0 instead of a

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return 0

testcases = [
    (48, 18, 6),
    (18, 48, 6),
    (7, 3, 1),
    (5, 0, 5),   #this test case will fail
]

for x, y, expected in testcases:
    got = gcd(x, y)
    smart_assert(got == expected)

