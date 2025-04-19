# test_factorial.py

from jaclang.runtimelib.gins.smart_assert import smart_assert

testcases = [
    (0, 1),
    (1, 1),
    (3, 6),
    (5, 120),
    (6, 720),
]

for idx, (inp, expected) in enumerate(testcases, 1):
    result = 0
    for i in range(1, inp + 1):
        result *= i
    got = result

