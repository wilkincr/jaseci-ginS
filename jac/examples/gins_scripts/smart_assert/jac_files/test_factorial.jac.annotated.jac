import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb2 → bb3 [exec=0, time=0.0000s] | bb1 → bb3 [exec=0, time=0.0000s] | bb0 → bb2, bb1 [exec=0, time=0.0000s]
import:py from examples.gins_scripts.smart_assert.python_files.algorithms, factorial ;  # bb4 → bb6 [exec=0, time=0.0000s] | bb3 → bb5, bb4 [exec=0, time=0.0000s] | bb5 → bb6 [exec=0, time=0.0000s]

with entry {

    testcases = [(0, 1), (1, 1), (3, 6), (5, 120), (6, 720)];      # bb6 → bb7 [exec=0, time=0.0000s]
    for (idx, (inp, expected)) in enumerate(testcases, 1) {  # bb8 [exec=0, time=0.0000s] | bb7 → bb8, bb7 [exec=0, time=0.0000s] | bb6 → bb7 [exec=0, time=0.0000s]

        got = factorial(inp);      # bb7 → bb8, bb7 [exec=0, time=0.0000s]
        smart_assert(got == expected) ;  # bb7 → bb8, bb7 [exec=0, time=0.0000s]
    }
}
