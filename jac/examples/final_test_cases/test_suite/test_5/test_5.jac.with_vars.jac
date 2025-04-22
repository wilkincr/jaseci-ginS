

with entry {
#   Variable values in this block:   s1 = ABCBDAB (seen 12×)  s2 = BDCABA (seen 11×)  m = 7 (seen 10×)  n = 6 (seen 9×)
    s1:str = 'ABCBDAB';
    s2:str = 'BDCABA';
    m:int = len(s1);
    n:int = len(s2);
    dp:list[list[int]] = [];
    for i in range(m + 1) {
        row:list[int] = [];
        #   Variable values in this block:   s1 = ABCBDAB (seen 64×)  s2 = BDCABA (seen 64×)  m = 7 (seen 64×)  n = 6 (seen 64×)
        for j in range(n + 1) {
            row.append(0);
            #   Variable values in this block:   s1 = ABCBDAB (seen 49×)  s2 = BDCABA (seen 49×)  m = 7 (seen 49×)  n = 6 (seen 49×)
        }
        dp.append(row);
        #   Variable values in this block:   s1 = ABCBDAB (seen 7×)  s2 = BDCABA (seen 7×)  m = 7 (seen 7×)  n = 6 (seen 7×)
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