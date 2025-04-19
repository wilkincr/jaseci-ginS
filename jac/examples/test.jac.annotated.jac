# Jaclang equivalent code
with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.000 ms
    n = 5; # BB: 0
    if (n / 0 == 5) { # BB: 0
        std.out("True");  # Use std.out for printing in Jaclang # BB: 1 Execution frequency: 0 Total execution time: 0.000 ms
    } else {
        std.out("False"); # BB: 2 Execution frequency: 0 Total execution time: 0.000 ms
    }
}