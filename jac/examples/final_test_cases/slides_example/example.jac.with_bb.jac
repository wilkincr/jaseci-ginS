with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
    x = 10; # BB: 0
    y = 0;     # BB: 0
    for i in range(10) { # BB: 0
        x = (x - 1); # BB: 1 Execution frequency: 11 Total execution time: 0.003 ms
        if (x > 0) { # BB: 1
            y = 0; # BB: 3 Execution frequency: 9 Total execution time: 0.000 ms
        }
    }
    y = (1 / x); # BB: 4 Execution frequency: 1 Total execution time: 0.000 ms
}