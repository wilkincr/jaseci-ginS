# Jaclang equivalent code

with entry {  # bb0 → bb2, bb1
        #   [LOAD_CONST 0 [jump_t=False] [arg=0,], LOAD_CONST ('annotations' [jump_t=False] [arg=1,], IMPORT_NAME __future__ [jump_t=False] [arg=0,], IMPORT_FROM annotations [jump_t=False] [arg=1,], STORE_NAME annotations [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
    n = 5;  # bb0 → bb2, bb1
        #   [LOAD_CONST 5 [jump_t=False] [arg=2,], STORE_NAME n [jump_t=False] [arg=2,]]

    if (n / 0 == 5) {  # bb0 → bb2, bb1
        #   [LOAD_NAME n [jump_t=False] [arg=2,], LOAD_CONST 0 [jump_t=False] [arg=0,], BINARY_OP / [jump_t=False] [arg=11,], LOAD_CONST 5 [jump_t=False] [arg=2,], COMPARE_OP == [jump_t=False] [arg=40,], POP_JUMP_IF_FALSE to 70 [jump_t=False] [arg=18,]]
        std.out("True");  # Use std.out for printing in Jaclang  # bb1 → bb2
        #   [LOAD_NAME std [jump_t=False] [arg=3,], LOAD_ATTR NULL|self + out [jump_t=False] [arg=9,], LOAD_CONST 'True' [jump_t=False] [arg=3,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,], RETURN_CONST None [jump_t=False] [arg=5,]]
    } else {
        std.out("False");  # bb2
        #   [LOAD_NAME std [jump_t=True] [arg=3,], LOAD_ATTR NULL|self + out [jump_t=False] [arg=9,], LOAD_CONST 'False' [jump_t=False] [arg=4,], CALL 1 [jump_t=False] [arg=1,], POP_TOP [jump_t=False] [arg=None,]]
    }
}