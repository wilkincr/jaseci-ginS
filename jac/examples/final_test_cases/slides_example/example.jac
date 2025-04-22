with entry {
    x:int = 10;
    y:int = 0;    
    for i in range(10) {
        x = (x - 1);
        if (x > 0) {
            y = 0;
        }
    }
    y = (1 / x);
}
