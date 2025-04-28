with entry {  # BB: 0 Execution frequency: 0 Total execution time: 0.000 ms
    x:int = 0; # BB: 0
    y:int = 3; # BB: 0
    z:int = x + y; # BB: 0
    for i in range(49) { # BB: 0
        x = 4 * i + y * (z); # BB: 1 Execution frequency: 0 Total execution time: 0.000 ms
        if x % 2 { # BB: 1
            y= 0; # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
        }
        else {
          y = 4; # BB: 3 Execution frequency: 0 Total execution time: 0.000 ms
        }
    }
    z = x/y; # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
    print(y); # BB: 4
    print("hello"); # BB: 4
}