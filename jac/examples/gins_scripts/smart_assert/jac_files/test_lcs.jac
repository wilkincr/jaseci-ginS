import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

can lcs(s1: str, s2: str) -> str {

    (m, n) = (len(s1), len(s2));    
    dp = [([0] * (n + 1)) for _ in range((m + 1))];
    
    
    for i in range(m) {
        for j in range(n) {

            if s1[i] == s2[j] {

                dp[(i + 1)][(j + 1)] = (dp[i][j] + 1);
            } else {

                dp[(i + 1)][(j + 1)] = max(dp[i][(j + 1)], dp[(i + 1)][j]);
            }
        }
    }
    
    (i, j) = (m, n);    
    seq = [];
    
    
    while i > 0 and j > 0 {

        if s1[(i - 1)] == s2[(j - 1)] {

            seq.append(s1[(i - 1)]) ;
            
            i -= 1;    
            j -= 1;
        } elif dp[(i - 1)][j] >= dp[i][(j - 1)] {

            i -= 1;
        } else {

            j -= 1;
        }
    }
    
    return ''.join(seq) ;
}

with entry {

    smart_assert(lcs('ABCBDAB', 'BDCABA') == 'BCBA') ;    
    smart_assert(lcs('XMJYAUZ', 'MZJAWXU') == 'MJAU') ;    
    smart_assert(lcs('', 'ANYTHING') == '') ;
}

