

with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.003 ms
    s1:str = 'ABCBDAB'; # BB: 0
    s2:str = 'BDCABA'; # BB: 0
    m:int = len(s1); # BB: 0
    n:int = len(s2); # BB: 0
    dp:list[list[int]] = []; # BB: 0
    for i in range(m + 1) { # BB: 0
        row:list[int] = []; # BB: 1 Execution frequency: 2 Total execution time: 0.001 ms
        for j in range(n + 1) { # BB: 1
            row.append(0); # BB: 2 Execution frequency: 2 Total execution time: 0.003 ms
        }
        dp.append(row); # BB: 3 Execution frequency: 1 Total execution time: 0.000 ms
    }
  
   
    for i in range(m) { # BB: 4 Execution frequency: 0 Total execution time: 0.000 ms
        for j in range(n) { # BB: 5 Execution frequency: 0 Total execution time: 0.000 ms
            if s1[i] == s2[j] { # BB: 6 Execution frequency: 0 Total execution time: 0.000 ms
                dp[(i + 1)][(j + 1)]:int = (dp[i][j] + 1); # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
            } else {
                dp[(i + 1)][(j + 1)]:int = max(dp[i][(j + 1)], dp[(i + 1)][j]); # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
            }
        }
    }
    i:int = m; # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
    j:int = n; # BB: 10
   
    seq:list[str] = []; # BB: 10
   
   
    while i > 0 and j > 0 { # BB: 10
        if s1[(i - 1)] == s2[(j - 1)] { # BB: 12 Execution frequency: 0 Total execution time: 0.000 ms
            seq.append(s1[(i - 1)]); # BB: 13 Execution frequency: 0 Total execution time: 0.000 ms
           
            i -= 1;     # BB: 13
            j -= 1; # BB: 13
        } elif dp[(i - 1)][j] >= dp[i][(j - 1)] { # BB: 14 Execution frequency: 0 Total execution time: 0.000 ms
            i -= 1; # BB: 15 Execution frequency: 0 Total execution time: 0.000 ms
        } else {
            j -= 1; # BB: 16 Execution frequency: 0 Total execution time: 0.000 ms
        }
    }
   
    print(reversed(seq)); # BB: 20 Execution frequency: 0 Total execution time: 0.000 ms
}