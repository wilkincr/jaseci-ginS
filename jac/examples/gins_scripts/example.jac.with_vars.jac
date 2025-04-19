with entry { 
#   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
    x:int = 0;
    #   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
    y:int = 3;
    #   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
    z:int = x + y;
    #   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
    for i in range(49) {
    #   Variable values in this block:   x = 0 (seen 3×)  x = 9 (seen 1×)  x = 4 (seen 1×)  x = 20 (seen 1×)  x = 24 (seen 1×)  y = 4 (seen 48×)  y = 3 (seen 2×)  y = 0 (seen 1×)  z = 3 (seen 50×)
        x = 4 * i + y * (z);
        #   Variable values in this block:   x = 9 (seen 2×)  x = 4 (seen 2×)  x = 20 (seen 2×)  x = 24 (seen 2×)  x = 28 (seen 2×)  y = 4 (seen 94×)  y = 3 (seen 2×)  y = 0 (seen 2×)  z = 3 (seen 98×)
        if x % 2 {
        #   Variable values in this block:   x = 9 (seen 2×)  x = 4 (seen 2×)  x = 20 (seen 2×)  x = 24 (seen 2×)  x = 28 (seen 2×)  y = 4 (seen 94×)  y = 3 (seen 2×)  y = 0 (seen 2×)  z = 3 (seen 98×)
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
    #   Variable values in this block:   x = 204 (seen 3×)  y = 4 (seen 3×)  z = 51.0 (seen 2×)  z = 3 (seen 1×)
    print("hello");
    #   Variable values in this block:   x = 204 (seen 3×)  y = 4 (seen 3×)  z = 51.0 (seen 2×)  z = 3 (seen 1×)
}