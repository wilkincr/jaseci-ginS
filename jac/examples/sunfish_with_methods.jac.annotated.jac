import:py from __future__, print_function ;  # bb0 → bb2, bb1 [exec=1, time=0.0039s] [mem=77580 bytes] | bb2 → bb3 [exec=1, time=0.0034s] [mem=77580 bytes] | bb1 → bb3 [exec=0, time=0.0000s] [mem=77580 bytes]

import:py time, math ;  # bb3 → bb5, bb4 [exec=1, time=0.0006s] [mem=77580 bytes] | bb4 → bb6 [exec=0, time=0.0000s] [mem=77580 bytes] | bb5 → bb6 [exec=1, time=0.0060s] [mem=77580 bytes]
import:py from itertools, count ;  # bb8 → bb9 [exec=1, time=0.0033s] [mem=77580 bytes] | bb6 → bb8, bb7 [exec=1, time=0.0006s] [mem=77580 bytes] | bb7 → bb9 [exec=0, time=0.0000s] [mem=77580 bytes]
import:py from collections, namedtuple, defaultdict ;  # bb9 → bb11, bb10 [exec=1, time=0.0006s] [mem=77580 bytes] | bb10 → bb12 [exec=0, time=0.0000s] [mem=77580 bytes] | bb11 → bb12 [exec=1, time=0.0038s] [mem=77580 bytes]

with entry {

    version = 'sunfish 2023';      # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
    piece = {'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000};      # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
    pst = {  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
        'P': (0, 0, 0, 0, 0, 0, 0, 0, 78, 83, 86, 73, 102, 82, 85, 90, 7, 29, 21, 44, 40, 31, 44, 7, -17, 16, -2, 15, 14, 0, 15, -13, -26, 3, 10, 9, 6, 1, 0, -23, -22, 9, 5, -11, -10, -2, 3, -19, -31, 8, -7, -37, -36, -14, 3, -31, 0, 0, 0, 0, 0, 0, 0, 0),  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
        'N': (-66, -53, -75, -75, -10, -55, -58, -70, -3, -6, 100, -36, 4, 62, -4, -14, 10, 67, 1, 74, 73, 27, 62, -2, 24, 24, 45, 37, 33, 41, 25, 17, -1, 5, 31, 21, 22, 35, 2, 0, -18, 10, 13, 22, 18, 15, 11, -14, -23, -15, 2, 0, 2, 0, -23, -20, -74, -23, -26, -24, -19, -35, -22, -69),  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
        'B': (-59, -78, -82, -76, -23, -107, -37, -50, -11, 20, 35, -42, -39, 31, 2, -22, -9, 39, -32, 41, 52, -10, 28, -14, 25, 17, 20, 34, 26, 25, 15, 10, 13, 10, 17, 23, 17, 16, 0, 7, 14, 25, 24, 15, 8, 25, 20, 15, 19, 20, 11, 6, 7, 6, 20, 16, -7, 2, -15, -12, -14, -15, -10, -10),  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
        'R': (35, 29, 33, 4, 37, 33, 56, 50, 55, 29, 56, 67, 55, 62, 34, 60, 19, 35, 28, 33, 45, 27, 25, 15, 0, 5, 16, 13, 18, -4, -9, -6, -28, -35, -16, -21, -13, -29, -46, -30, -42, -28, -42, -25, -25, -35, -26, -46, -53, -38, -31, -26, -29, -43, -44, -53, -30, -24, -18, 5, -2, -18, -31, -32),  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
        'Q': (6, 1, -8, -104, 69, 24, 88, 26, 14, 32, 60, -10, 20, 76, 57, 24, -2, 43, 32, 60, 72, 63, 43, 2, 1, -16, 22, 17, 25, 20, -13, -6, -14, -15, -2, -5, -1, -10, -20, -22, -30, -6, -13, -11, -16, -11, -16, -27, -36, -18, 0, -19, -15, -15, -21, -38, -39, -30, -31, -13, -31, -36, -34, -42),  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
        'K': (4, 54, 47, -99, -99, 60, 83, -62, -32, 10, 55, 56, 56, 55, 10, 3, -62, 12, -57, 44, -67, 28, 37, -31, -55, 50, 11, -4, -19, 13, 0, -49, -55, -43, -52, -28, -51, -47, -8, -50, -47, -42, -43, -79, -64, -32, -29, -32, -4, 3, -14, -50, -57, -18, 13, 4, 17, 30, -3, -14, 6, -1, 40, 18)  # bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes]
    };    
    for (k, table) in pst.items() {  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes] | bb12 → bb13 [exec=1, time=0.0071s] [mem=77580 bytes] | bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]

        padrow = with row: Any can (((0  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes]
        ,) + <>tuple(((x + piece[k])  for x in row))) + (0
        ,));
        

        pst[k] = sum(  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes]
            (padrow(table[(i * 8):((i * 8) + 8)])  for i in range(8)),  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes]
            ()  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes]
        );
        

        pst[k] = ((((0  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes]
        ,) * 20) + pst[k]) + ((0  # bb13 → bb14, bb13 [exec=1, time=2.2582s] [mem=77580 bytes]
        ,) * 20));
    }

    
    (A1, H1, A8, H8) = (91, 98, 21, 28);      # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    initial = '         \n         \n rnbqkbnr\n pppppppp\n ........\n ........\n ........\n ........\n PPPPPPPP\n RNBQKBNR\n         \n         \n'  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    ;    
    (N, E, S, W) = (-10, 1, 10, -1);      # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    directions = {  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        'P': (N, (N + N)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(N + W)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(N + E)),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        'N': (((N + N) + E), ((E + N) + E), ((E + S) + E), ((S + S) + E), ((S + S) + W), ((W + S) + W), ((W + N) + W), ((N + N) + W)),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        'B': ((N + E), (S + E), (S + W), (N + W)),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        'R': (N, E, S, W),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        'Q': (N,   # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        E,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        S,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        W,(N + E)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(S + E)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(S + W)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(N + W)),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        'K': (N,   # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        E,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        S,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        W,(N + E)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(S + E)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(S + W)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        ,(N + W))  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    };
    
    MATE_LOWER = (piece['K'] - (10 * piece['Q']));  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    
    MATE_UPPER = (piece['K'] + (10 * piece['Q']));  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    
    QS = 40;      # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    QS_A = 140;      # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    EVAL_ROUGHNESS = 15;      # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    opt_ranges = <>dict(  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        QS=(0, 300),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        QS_A=(0, 300),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        EVAL_ROUGHNESS=(0, 50)  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    );    
    Move = namedtuple('Move', 'i j prom');  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
}

"""A state of a chess game\n    board -- a 120 char representation of the board\n    score -- the board evaluation\n    wc -- the castling rights, [west/queen side, east/king side]\n    bc -- the opponent castling rights, [west/king side, east/queen side]\n    ep - the en passant square\n    kp - the king passant square\n    """  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
class Position : namedtuple('Position', 'board score wc bc ep kp') : {  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    can gen_moves() {
        for (i, p) in enumerate(self.board) {
            if not p.isupper() {
                continue ;
            }
            
            for d in directions[p] {
                for j in count((i + d), d) {

                    q = self.board[j];    
                    if q.isspace() or q.isupper() {
                        break ;
                    }

                    
                    if p == 'P' {
                        if d in (N, (N + N)) and q != '.' {
                            break ;
                        }
                        
                        if d == (N + N)
                            and i < (A1 + N) or self.board[(i + N)] != '.' {
                            break ;
                        }
                        
                        if d in ((N + W), (N + E))
                            and q == '.'
                            and j not in (self.ep, 
                        self.kp,(self.kp - 1)
                        ,(self.kp + 1)) {
                            break ;
                        }
                        
                        if A8 <= j<= H8 {
                            for prom in 'NBRQ' {

                                yield Move(i, j, prom);
                                 ;
                            }
                            
                            break ;
                        }
                    }

                    
                    yield Move(i, j, '');
                     ;    
                    if p in 'PNK' or q.islower() {
                        break ;
                    }

                    
                    if i == A1
                        and self.board[(j + E)] == 'K'
                        and self.wc[0] {

                        yield Move((j + E), (j + W), '');
                         ;
                    }
                    
                    if i == H1
                        and self.board[(j + W)] == 'K'
                        and self.wc[1] {

                        yield Move((j + W), (j + E), '');
                         ;
                    }
                }
            }
        }
    }
    
    can rotate(nullmove: Any=False) {

        """Rotates the board, preserving enpassant, unless nullmove""" ;    
        return Position(
            self.board[::-1].swapcase(),
            -self.score,
            self.bc,
            self.wc,
            (119 - self.ep) if self.ep and not nullmove else 0,
            (119 - self.kp) if self.kp and not nullmove else 0
        ) ;
    }
    
    can move(move: Any) {

        (i, j, prom) = move;    
        (p, q) = (self.board[i], self.board[j]);    
        put = with board: Any, i: Any, p: Any can ((board[:i] + p) + board[(i + 1):]);
        
        board = self.board;    
        (wc, bc, ep, kp) = (self.wc, self.bc, 0, 0);    
        score = (self.score + self.value(move));
        
        board = put(board, j, board[i]);    
        board = put(board, i, '.');    
        if i == A1 {

            wc = (False, wc[1]);
        }
        
        if i == H1 {

            wc = (wc[0], False);
        }
        
        if j == A8 {

            bc = (bc[0], False);
        }
        
        if j == H8 {

            bc = (False, bc[1]);
        }
        
        if p == 'K' {

            wc = (False, False);    
            if <>abs((j - i)) == 2 {

                kp = ((i + j) // 2);
                
                board = put(board, A1 if j < i else H1, '.');    
                board = put(board, kp, 'R');
            }
        }
        
        if p == 'P' {
            if A8 <= j<= H8 {

                board = put(board, j, prom);
            }
            
            if (j - i) == (2 * N) {

                ep = (i + N);
            }

            
            if j == self.ep {

                board = put(board, (j + S), '.');
            }
        }
        
        return Position(board, score, wc, bc, ep, kp).rotate() ;
    }
    
    can value(move: Any) {

        (i, j, prom) = move;    
        (p, q) = (self.board[i], self.board[j]);    
        score = (pst[p][j] - pst[p][i]);
        
        if q.islower() {

            score += pst[q.upper()][(119 - j)];
        }

        
        if <>abs((j - self.kp)) < 2 {

            score += pst['K'][(119 - j)];
        }

        
        if p == 'K' and <>abs((i - j)) == 2 {

            score += pst['R'][((i + j) // 2)];
            
            score -= pst['R'][A1 if j < i else H1];
        }
        
        if p == 'P' {
            if A8 <= j<= H8 {

                score += (pst[prom][j] - pst['P'][j]);
            }

            
            if j == self.ep {

                score += pst['P'][(119 - (j + S))];
            }
        }

        
        return score ;
    }
}

with entry {

    Entry = namedtuple('Entry', 'lower upper');  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
}

class Searcher {  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    can init() {

        self.tp_score = {};
        
        self.tp_move = {};
        
        self.history = <>set();    
        self.nodes = 0;
    }
    
    can bound(pos: Any, gamma: Any, depth: Any, can_null: Any=True) {

        """ Let s* be the "true" score of the sub-tree we are searching.\n            The method returns r, where\n            if gamma >  s* then s* <= r < gamma  (A better upper bound)\n            if gamma <= s* then gamma <= r <= s* (A better lower bound) """ ;    
        self.nodes += 1;    

        depth = max(depth, 0);    
        if pos.score <= -MATE_LOWER {
            return -MATE_UPPER ;
        }
        

        <>entry = self.tp_score.get(
            (pos, depth, can_null),
            Entry(-MATE_UPPER, MATE_UPPER)
        );    
        if <>entry.lower >= gamma {

            return <>entry.lower ;
        }
        
        if <>entry.upper < gamma {

            return <>entry.upper ;
        }
        
        if can_null
            and depth > 0
            and pos in self.history {

            return 0 ;
        }
        
        can moves() {
            if depth > 2
                and can_null
                and <>abs(pos.score) < 500 {

                yield (None, -self.bound(
                    pos.rotate(nullmove=True),
                    (1 - gamma),
                    (depth - 3)
                ));
                 ;
            }
            
            if depth == 0 {

                yield (None, pos.score);
                 ;
            }
            

            killer = self.tp_move.get(pos);    
            if not killer and depth > 2 {

                self.bound(pos, gamma, (depth - 3), can_null=False) ;    
                killer = self.tp_move.get(pos);
            }
            

            val_lower = (QS - (depth * QS_A));    
            if killer and pos.value(killer) >= val_lower {

                yield (killer, -self.bound(pos.move(killer), (1 - gamma), (depth - 1)));
                 ;
            }
            
            for (val, move) in sorted(
                ((pos.value(m), m)  for m in pos.gen_moves()),
                reverse=True
            ) {
                if val < val_lower {
                    break ;
                }

                
                if depth <= 1 and (pos.score + val) < gamma {

                    yield (move, (pos.score + val) if val < MATE_LOWER else MATE_UPPER);
                     ;    
                    break ;
                }

                

                yield (move, -self.bound(pos.move(move), (1 - gamma), (depth - 1)));
                 ;
            }
        }

        

        best = -MATE_UPPER;    
        for (move, score) in moves() {

            best = max(best, score);    
            if best >= gamma {
                if move is not None {

                    self.tp_move[pos] = move;
                }
                
                break ;
            }
        }

        
        if depth > 2 and best == -MATE_UPPER {

            flipped = pos.rotate(nullmove=True);    
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER;    
            best = -MATE_LOWER if in_check else 0;
        }
        
        if best >= gamma {

            self.tp_score[(pos, depth, can_null)] = Entry(best, <>entry.upper);
        }
        
        if best < gamma {

            self.tp_score[(pos, depth, can_null)] = Entry(<>entry.lower, best);
        }
        

        return best ;
    }
    
    can search(history: Any) {

        """Iterative deepening MTD-bi search""" ;    
        self.nodes = 0;    

        self.history = <>set(history);    

        self.tp_score.clear() ;    

        gamma = 0;    
        for depth in range(1, 1000) {

            (lower, upper) = (-MATE_LOWER, MATE_LOWER);    
            while lower < (upper - EVAL_ROUGHNESS) {

                score = self.bound(history[-1], gamma, depth, can_null=False);    
                if score >= gamma {

                    lower = score;
                }
                
                if score < gamma {

                    upper = score;
                }
                

                yield (depth, gamma, score, self.tp_move.get(history[-1]));
                 ;    

                gamma = (((lower + upper) + 1) // 2);
            }
        }
    }
}

can parse(c: Any) {  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]

    (fil, rank) = ((ord(c[0]) - ord('a')), (int(c[1]) - 1));
    
    return ((A1 + fil) - (10 * rank)) ;
}

can render(i: Any) {  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]

    (rank, fil) = divmod((i - A1), 10);    
    return (chr((fil + ord('a'))) + str((-rank + 1))) ;
}

with entry {

    hist = [Position(  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        initial,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        0,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        (True, True),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        (True, True),  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        0,  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
        0  # bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
    )];
}

import:py sys, tools.uci ;  # bb16 → bb17 [exec=1, time=0.0000s] [mem=77580 bytes] | bb15 → bb17 [exec=0, time=0.0000s] [mem=77580 bytes] | bb14 → bb16, bb15 [exec=1, time=0.2110s] [mem=77580 bytes]
with entry {

    tools.uci.run(sys.modules[__name__], hist[-1]) ;      # bb17 → bb18 [exec=0, time=0.0000s] [mem=77580 bytes]
    sys.exit() ;      # bb17 → bb18 [exec=0, time=0.0000s] [mem=77580 bytes]
    searcher = Searcher();      # bb17 → bb18 [exec=0, time=0.0000s] [mem=77580 bytes]
    while True {  # bb17 → bb18 [exec=0, time=0.0000s] [mem=77580 bytes] | bb49 → bb18 [exec=0, time=0.0000s] [mem=77580 bytes] | bb50 [exec=0, time=0.0000s] [mem=77580 bytes]

        args = input().split();      # bb18 → bb20, bb19 [exec=0, time=0.0000s] [mem=77580 bytes]
        if args[0] == 'uci' {  # bb18 → bb20, bb19 [exec=0, time=0.0000s] [mem=77580 bytes]

            print('id name', version) ;      # bb19 → bb49 [exec=0, time=0.0000s] [mem=77580 bytes]
            print('uciok') ;  # bb19 → bb49 [exec=0, time=0.0000s] [mem=77580 bytes]
        } elif args[0] == 'isready' {  # bb20 → bb22, bb21 [exec=0, time=0.0000s] [mem=77580 bytes]

            print('readyok') ;  # bb21 → bb49 [exec=0, time=0.0000s] [mem=77580 bytes]
        } elif args[0] == 'quit' {  # bb22 → bb24, bb23 [exec=0, time=0.0000s] [mem=77580 bytes]
            break ;  # bb23 → bb24 [exec=0, time=0.0000s] [mem=77580 bytes]
        } elif args[:2] == ['position', 'startpos'] {  # bb24 → bb30, bb25 [exec=0, time=0.0000s] [mem=77580 bytes]

            del (hist[1:]  # bb25 → bb26 [exec=0, time=0.0000s] [mem=77580 bytes]
            ,) ;    
            for (ply, move) in enumerate(args[3:]) {  # bb29 → bb49 [exec=0, time=0.0000s] [mem=77580 bytes] | bb25 → bb26 [exec=0, time=0.0000s] [mem=77580 bytes] | bb26 → bb29, bb28, bb27 [exec=0, time=0.0000s] [mem=77580 bytes]

                (i, j, prom) = (parse(move[:2]), parse(move[2:4]), move[4:].upper());      # bb26 → bb29, bb28, bb27 [exec=0, time=0.0000s] [mem=77580 bytes]
                if (ply % 2) == 1 {  # bb26 → bb29, bb28, bb27 [exec=0, time=0.0000s] [mem=77580 bytes]

                    (i, j) = ((119 - i), (119 - j));  # bb27 → bb28 [exec=0, time=0.0000s] [mem=77580 bytes]
                }

                
                hist.append(hist[-1].move(Move(i, j, prom))) ;  # bb28 → bb26 [exec=0, time=0.0000s] [mem=77580 bytes]
            }
        } elif args[0] == 'go' {  # bb30 → bb49, bb31 [exec=0, time=0.0000s] [mem=77580 bytes]

            (wtime, btime, winc, binc) = [(int(a) / 1000) for a in args[2::2]];      # bb31 → bb32 [exec=0, time=0.0000s] [mem=77580 bytes] | bb32 → bb33, bb32 [exec=0, time=0.0000s] [mem=77580 bytes] | bb50 [exec=0, time=0.0000s] [mem=77580 bytes] | bb33 → bb35, bb34 [exec=0, time=0.0000s] [mem=77580 bytes]
            if (len(hist) % 2) == 0 {  # bb33 → bb35, bb34 [exec=0, time=0.0000s] [mem=77580 bytes]

                (wtime, winc) = (btime, binc);  # bb34 → bb35 [exec=0, time=0.0000s] [mem=77580 bytes]
            }
            
            think = min(((wtime / 40) + winc), ((wtime / 2) - 1));  # bb35 → bb36 [exec=0, time=0.0000s] [mem=77580 bytes]
            
            start = time.time();      # bb35 → bb36 [exec=0, time=0.0000s] [mem=77580 bytes]
            move_str = None;      # bb35 → bb36 [exec=0, time=0.0000s] [mem=77580 bytes]
            for (depth, gamma, score, move) in Searcher().search(hist) {  # bb45 → bb46 [exec=0, time=0.0000s] [mem=77580 bytes] | bb35 → bb36 [exec=0, time=0.0000s] [mem=77580 bytes] | bb36 → bb45, bb40, bb37 [exec=0, time=0.0000s] [mem=77580 bytes]
                if score >= gamma {  # bb36 → bb45, bb40, bb37 [exec=0, time=0.0000s] [mem=77580 bytes]

                    (i, j) = (move.i, move.j);      # bb37 → bb39, bb38 [exec=0, time=0.0000s] [mem=77580 bytes]
                    if (len(hist) % 2) == 0 {  # bb37 → bb39, bb38 [exec=0, time=0.0000s] [mem=77580 bytes]

                        (i, j) = ((119 - i), (119 - j));  # bb38 → bb39 [exec=0, time=0.0000s] [mem=77580 bytes]
                    }

                    
                    move_str = ((render(i) + render(j)) + move.prom.lower());  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                    
                    print(  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                        'info depth',  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                        depth,  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                        'score cp',  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                        score,  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                        'pv',  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                        move_str  # bb39 → bb40 [exec=0, time=0.0000s] [mem=77580 bytes]
                    ) ;
                }
                
                if move_str  # bb41 → bb36 [exec=0, time=0.0000s] [mem=77580 bytes] | bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=77580 bytes]
                    and (time.time() - start) > (think * 0.8) {  # bb42 → bb44, bb43 [exec=0, time=0.0000s] [mem=77580 bytes] | bb43 → bb36 [exec=0, time=0.0000s] [mem=77580 bytes]
                    break ;  # bb44 → bb46 [exec=0, time=0.0000s] [mem=77580 bytes]
                }
            }

            
            print('bestmove', move_str or '(none)') ;  # bb46 → bb48, bb47 [exec=0, time=0.0000s] [mem=77580 bytes] | bb48 → bb49 [exec=0, time=0.0000s] [mem=77580 bytes] | bb47 → bb48 [exec=0, time=0.0000s] [mem=77580 bytes]
        }
    }
}
