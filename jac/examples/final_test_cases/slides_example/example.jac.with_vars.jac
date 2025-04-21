with entry {
#   Variable values in this block:   x = 10 (seen 2×)  x = 9 (seen 1×)  x = 8 (seen 1×)  x = 7 (seen 1×)  x = 6 (seen 1×)  y = 0 (seen 11×)

    x:int = 10;
    y:int = 0;    
    for i in range(10) {

        x = (x - 1);
        #   Variable values in this block:   x = 10 (seen 1×)  x = 9 (seen 1×)  x = 8 (seen 1×)  x = 7 (seen 1×)  x = 6 (seen 1×)  y = 0 (seen 10×)
    }
    
    y = (1 / x);
    #   Variable values in this block:   x = 0 (seen 1×)  y = 0 (seen 1×)
}
