with entry { 
    x:int = 0;
    y:int = 3;
    z:int = x + y;
    for i in range(49) {
        x = 4 * i + y * (z);
        if x % 2 {
            y= 0;
        }
        else {
          y = 4;
        }
    }
    z = x/y;
    print(y);
    print("hello");
}