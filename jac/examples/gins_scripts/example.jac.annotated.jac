with entry {   # bb0 → bb1 [exec=1, time=0.0012s] [mem=121776 bytes]
    x:int = 0;  # bb0 → bb1 [exec=1, time=0.0012s] [mem=121776 bytes]
    y:int = 3;  # bb0 → bb1 [exec=1, time=0.0012s] [mem=121776 bytes]
    z:int = x + y;  # bb0 → bb1 [exec=1, time=0.0012s] [mem=121776 bytes]
    for i in range(49) {  # bb1 → bb4, bb3, bb2 [exec=50, time=0.0784s] [mem=121776 bytes] | bb0 → bb1 [exec=1, time=0.0012s] [mem=121776 bytes] | bb4 [exec=1, time=0.0000s] [mem=121776 bytes]
        x = 4 * i + y * (z);  # bb1 → bb4, bb3, bb2 [exec=50, time=0.0784s] [mem=121776 bytes]
        if x % 2 {  # bb1 → bb4, bb3, bb2 [exec=50, time=0.0784s] [mem=121776 bytes]
            y= 0;  # bb2 → bb1 [exec=1, time=0.0001s] [mem=121776 bytes]
        }
        else {
          y = 4;  # bb3 → bb1 [exec=48, time=0.0115s] [mem=121776 bytes]
        }
    }
    z = x/y;  # bb4 [exec=1, time=0.0000s] [mem=121776 bytes]
    print(y);  # bb4 [exec=1, time=0.0000s] [mem=121776 bytes]
    print("hello");  # bb4 [exec=1, time=0.0000s] [mem=121776 bytes]
}