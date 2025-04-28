with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.002 ms
    x:int = 10; # BB: 0
    y:int = 0;     # BB: 0
    for i in range(10) { # BB: 0
        x = (x - 1); # BB: 1 Execution frequency: 6 Total execution time: 0.005 ms
        if (x > 0) { # BB: 1
            y = 0; # BB: 3 Execution frequency: 6 Total execution time: 0.002 ms
        }
    }
    y = (1 / x); # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
}