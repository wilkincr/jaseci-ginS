import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;  # bb0 → bb2, bb1 [exec=0, time=0.0000s] [mem=38196 bytes] | bb1 → bb3 [exec=0, time=0.0000s] [mem=38196 bytes] | bb2 → bb3 [exec=0, time=0.0000s] [mem=38196 bytes]

with entry {

    x = 10;      # bb3 → bb4 [exec=0, time=0.0000s] [mem=38196 bytes]
    for i in range(10) {  # bb5 [exec=0, time=0.0000s] [mem=38196 bytes] | bb4 → bb5, bb4 [exec=0, time=0.0000s] [mem=38196 bytes] | bb3 → bb4 [exec=0, time=0.0000s] [mem=38196 bytes]

        x = (x - 1);  # bb4 → bb5, bb4 [exec=0, time=0.0000s] [mem=38196 bytes]
    }

    smart_assert(x != 0, 'test') ;      # bb5 [exec=0, time=0.0000s] [mem=38196 bytes]
    
    y = (1 / x);  # bb5 [exec=0, time=0.0000s] [mem=38196 bytes]
}
