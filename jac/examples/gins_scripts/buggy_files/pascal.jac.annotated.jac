with entry {  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]

    testcases = [  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]
        (1, [[1]]),  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]
        (2, [[1], [1, 1]]),  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]
        (3, [[1], [1, 1], [1, 2, 1]]),  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]
        (4, [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]),  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]
        (5, [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]])  # bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes]
    ];    
    for (i, (n, expected)) in enumerate(testcases) {  # bb1 → bb12, bb2 [exec=3, time=0.0020s] [mem=72580 bytes] | bb0 → bb1 [exec=1, time=0.0033s] [mem=72580 bytes] | bb12 [exec=0, time=0.0000s] [mem=72580 bytes]

        rows = [[1]];      # bb1 → bb12, bb2 [exec=3, time=0.0020s] [mem=72580 bytes]
        for r in range(1, n) {  # bb2 → bb11, bb3 [exec=5, time=0.0015s] [mem=72580 bytes] | bb1 → bb12, bb2 [exec=3, time=0.0020s] [mem=72580 bytes] | bb11 → bb1 [exec=2, time=0.0092s] [mem=72580 bytes]

            row = [];  # bb2 → bb11, bb3 [exec=5, time=0.0015s] [mem=72580 bytes]
            
            for c in range(0, r) {  # bb2 → bb11, bb3 [exec=5, time=0.0015s] [mem=72580 bytes] | bb3 → bb10, bb5, bb4 [exec=6, time=0.0012s] [mem=72580 bytes] | bb10 → bb2 [exec=2, time=0.0008s] [mem=72580 bytes]

                upleft = rows[(r - 1)][(c - 1)] if c > 0 else 0;      # bb5 → bb6 [exec=3, time=0.0001s] [mem=72580 bytes] | bb3 → bb10, bb5, bb4 [exec=6, time=0.0012s] [mem=72580 bytes] | bb4 → bb6 [exec=1, time=0.0005s] [mem=72580 bytes] | bb6 → bb8, bb7 [exec=4, time=0.0009s] [mem=72580 bytes]

                upright = rows[(r - 1)][c] if c < r else 0;      # bb8 → bb9 [exec=0, time=0.0000s] [mem=72580 bytes] | bb6 → bb8, bb7 [exec=4, time=0.0009s] [mem=72580 bytes] | bb7 → bb9 [exec=4, time=0.0066s] [mem=72580 bytes] | bb9 → bb3 [exec=3, time=0.0015s] [mem=72580 bytes]

                row.append((upleft + upright)) ;  # bb9 → bb3 [exec=3, time=0.0015s] [mem=72580 bytes]
            }

            
            rows.append(row) ;  # bb10 → bb2 [exec=2, time=0.0008s] [mem=72580 bytes]
        }
        
        print('Input:', n) ;      # bb11 → bb1 [exec=2, time=0.0092s] [mem=72580 bytes]
        print('Expected:', expected) ;      # bb11 → bb1 [exec=2, time=0.0092s] [mem=72580 bytes]
        print('Got     :', rows) ;      # bb11 → bb1 [exec=2, time=0.0092s] [mem=72580 bytes]
        print('Pass    :', rows == expected) ;      # bb11 → bb1 [exec=2, time=0.0092s] [mem=72580 bytes]
        print() ;  # bb11 → bb1 [exec=2, time=0.0092s] [mem=72580 bytes]
    }
}
