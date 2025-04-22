

with entry {
    s1:str = 'ABCBDAB';
    s2:str = 'BDCABA';
    m:int = len(s1);
    n:int = len(s2);
    dp:list[list[int]] = [];
    for i in range(m + 1) {
        row:list[int] = [];
        for j in range(n + 1) {
            row.append(0);
        }
        dp.append(row);
    }
  
   
    for i in range(m) {
        for j in range(n) {
            if s1[i] == s2[j] {
                dp[(i + 1)][(j + 1)]:int = (dp[i][j] + 1);
            } else {
                dp[(i + 1)][(j + 1)]:int = max(dp[i][(j + 1)], dp[(i + 1)][j]);
            }
        }
    }
    i:int = m;
    j:int = n;
   
    seq:list[str] = [];
   
   
    while i > 0 and j > 0 {
        if s1[(i - 1)] == s2[(j - 1)] {
            seq.append(s1[(i - 1)]);
           
            i -= 1;    
            j -= 1;
        } elif dp[(i - 1)][j] >= dp[i][(j - 1)] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
   
    print(reversed(seq));
}
