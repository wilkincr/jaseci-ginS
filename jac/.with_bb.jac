with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms

    x = 10;     # BB: 0
    for i in range(10) { # BB: 0

        x = (x - 1); # BB: 1 Execution frequency: 1 Total execution time: 0.001 ms
    }
    
    y = (1 / x); # BB: 2 Execution frequency: 1 Total execution time: 0.000 ms
}
