import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;
import:py from examples.gins_scripts.smart_assert.python_files.algorithms, factorial ;

with entry {

    testcases = [(0, 1), (1, 1), (3, 6), (5, 120), (6, 720)];    
    for (idx, (inp, expected)) in enumerate(testcases, 1) {

        got = factorial(inp);    
        smart_assert(got == expected) ;
    }
}

