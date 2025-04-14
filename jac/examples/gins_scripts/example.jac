with entry { 
    x:int = 0;
    y:int = 3;
    z:int = x + y;
    for i in range(49) {
        x = 4 * i + y * (z);
        if x % 2 == 0:
            y = 4;
        else:
            y = 0;
    }
    if y != 0:
        z = x / y;
    print(y);
    print("hello");
}
