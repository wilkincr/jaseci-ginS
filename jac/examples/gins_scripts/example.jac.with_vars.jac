with entry { 
#   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
    x:int = 0;
    y:int = 3;
    z:int = x + y;
    for i in range(49) {
        x = 4 * i + y * (z);
        #   Variable values in this block:   x = 9 (seen 2×)  x = 4 (seen 2×)  x = 20 (seen 2×)  x = 24 (seen 2×)  x = 28 (seen 2×)  y = 4 (seen 94×)  y = 3 (seen 2×)  y = 0 (seen 2×)  z = 3 (seen 98×)
        if x % 2 {
            y= 0;
            #   Variable values in this block:   x = 9 (seen 1×)  y = 3 (seen 1×)  z = 3 (seen 1×)
        }
        else {
          y = 4;
          #   Variable values in this block:   x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  x = 28 (seen 1×)  x = 32 (seen 1×)  y = 4 (seen 47×)  y = 0 (seen 1×)  z = 3 (seen 48×)
        }
    }
    z = x/y;
    #   Variable values in this block:   x = 204 (seen 3×)  y = 4 (seen 3×)  z = 51.0 (seen 2×)  z = 3 (seen 1×)
    print(y);
    print("hello");
}