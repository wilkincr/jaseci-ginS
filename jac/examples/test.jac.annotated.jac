# Jaclang equivalent code
with entry {  # bb0 → bb2, bb1 [exec=1, time=0.0000s]
    n = 5;  # bb0 → bb2, bb1 [exec=1, time=0.0000s]
    if (n / 0 == 5) {  # bb0 → bb2, bb1 [exec=1, time=0.0000s]
        std.out("True");  # Use std.out for printing in Jaclang  # bb1 → bb2 [exec=0, time=0.0000s]
    } else {
        std.out("False");  # bb2 [exec=0, time=0.0000s]
    }
}