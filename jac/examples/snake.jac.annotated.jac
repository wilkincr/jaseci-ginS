import:py curses ;  # bb2 → bb3 [exec=1, time=0.0029s] [mem=781349 bytes] | bb1 → bb3 [exec=0, time=0.0000s] [mem=781349 bytes] | bb0 → bb2, bb1 [exec=1, time=0.0035s] [mem=781349 bytes]
import:py time ;  # bb4 → bb6 [exec=0, time=0.0000s] [mem=781349 bytes] | bb5 → bb6 [exec=1, time=0.0026s] [mem=781349 bytes] | bb3 → bb5, bb4 [exec=1, time=0.0006s] [mem=781349 bytes]
import:py sys ;  # bb6 → bb8, bb7 [exec=1, time=0.0006s] [mem=781349 bytes] | bb8 → bb9 [exec=1, time=0.0026s] [mem=781349 bytes] | bb7 → bb9 [exec=0, time=0.0000s] [mem=781349 bytes]
import:py from random, randint ;  # bb11 → bb12 [exec=1, time=0.0032s] [mem=781349 bytes] | bb9 → bb11, bb10 [exec=1, time=0.0006s] [mem=781349 bytes] | bb10 → bb12 [exec=0, time=0.0000s] [mem=781349 bytes]

with entry {
    screen = curses.initscr();      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    curses.noecho() ;      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    curses.cbreak() ;      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    screen.keypad(True) ;      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    screen.timeout(0) ;      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    size = 10;      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    icons = {0: ' . ', 1: ' * ', 2: ' # ', 3: ' & '};      # bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes]
    field = [[0 for j in range(size)] for i in range(size)];      # bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes] | bb14 → bb15, bb14 [exec=10, time=0.1575s] [mem=781349 bytes] | bb15 → bb13 [exec=10, time=0.0082s] [mem=781349 bytes] | bb12 → bb13 [exec=1, time=0.0137s] [mem=781349 bytes] | bb13 → bb16, bb14 [exec=11, time=0.0219s] [mem=781349 bytes] | bb103 [exec=0, time=0.0000s] [mem=781349 bytes]
    snake_coords = [[0, 0], [0, 1], [0, 2], [0, 3]];      # bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes]
    direction = curses.KEY_RIGHT;      # bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes]
    i = randint(0, (size - 1));  # bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes]
    
    j = randint(0, (size - 1));  # bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes]
    
    entity = [i, j];      # bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes]
    while entity in snake_coords {  # bb18 → bb17 [exec=0, time=0.0000s] [mem=781349 bytes] | bb16 → bb19, bb17 [exec=1, time=0.0238s] [mem=781349 bytes] | bb17 → bb19, bb18 [exec=1, time=0.0048s] [mem=781349 bytes]

        i = randint(0, (size - 1));  # bb17 → bb19, bb18 [exec=1, time=0.0048s] [mem=781349 bytes]
        
        j = randint(0, (size - 1));  # bb17 → bb19, bb18 [exec=1, time=0.0048s] [mem=781349 bytes]
        
        entity = [i, j];  # bb17 → bb19, bb18 [exec=1, time=0.0048s] [mem=781349 bytes]
    }
    
    field[i][j] = 3;      # bb19 → bb20 [exec=1, time=0.0014s] [mem=781349 bytes]
    while True {  # bb102 → bb20 [exec=0, time=0.0000s] [mem=781349 bytes] | bb19 → bb20 [exec=1, time=0.0014s] [mem=781349 bytes] | bb103 [exec=0, time=0.0000s] [mem=781349 bytes]

        ch = screen.getch();      # bb20 → bb34, bb21 [exec=1, time=0.0017s] [mem=781349 bytes]
        if ch != -1 {  # bb20 → bb34, bb21 [exec=1, time=0.0017s] [mem=781349 bytes]

            invalid_direction = ch == curses.KEY_LEFT  # bb22 → bb23 [exec=0, time=0.0000s] [mem=781349 bytes] | bb24 → bb26, bb25 [exec=0, time=0.0000s] [mem=781349 bytes] | bb32 → bb34, bb33 [exec=0, time=0.0000s] [mem=781349 bytes] | bb27 → bb29, bb28 [exec=0, time=0.0000s] [mem=781349 bytes] | bb29 → bb32, bb30 [exec=0, time=0.0000s] [mem=781349 bytes] | bb21 → bb23, bb22 [exec=0, time=0.0000s] [mem=781349 bytes] | bb23 → bb32, bb24 [exec=0, time=0.0000s] [mem=781349 bytes] | bb26 → bb32, bb27 [exec=0, time=0.0000s] [mem=781349 bytes] | bb30 → bb32, bb31 [exec=0, time=0.0000s] [mem=781349 bytes]
                and direction == curses.KEY_RIGHT  # bb22 → bb23 [exec=0, time=0.0000s] [mem=781349 bytes]
                or ch == curses.KEY_RIGHT  # bb24 → bb26, bb25 [exec=0, time=0.0000s] [mem=781349 bytes] | bb25 → bb26 [exec=0, time=0.0000s] [mem=781349 bytes]
                and direction == curses.KEY_LEFT  # bb25 → bb26 [exec=0, time=0.0000s] [mem=781349 bytes]
                or ch == curses.KEY_UP  # bb28 → bb29 [exec=0, time=0.0000s] [mem=781349 bytes] | bb27 → bb29, bb28 [exec=0, time=0.0000s] [mem=781349 bytes]
                and direction == curses.KEY_DOWN  # bb28 → bb29 [exec=0, time=0.0000s] [mem=781349 bytes]
                or ch == curses.KEY_DOWN  # bb31 → bb32 [exec=0, time=0.0000s] [mem=781349 bytes] | bb30 → bb32, bb31 [exec=0, time=0.0000s] [mem=781349 bytes]
                and direction == curses.KEY_UP;      # bb31 → bb32 [exec=0, time=0.0000s] [mem=781349 bytes]
            if not invalid_direction {  # bb32 → bb34, bb33 [exec=0, time=0.0000s] [mem=781349 bytes]

                direction = ch;  # bb33 → bb34 [exec=0, time=0.0000s] [mem=781349 bytes]
            }
        }
        
        head = snake_coords[-1][:];  # bb34 → bb36, bb35 [exec=1, time=0.0025s] [mem=781349 bytes]
        
        if direction == curses.KEY_UP {  # bb34 → bb36, bb35 [exec=1, time=0.0025s] [mem=781349 bytes]

            head[0] -= 1;  # bb35 → bb42 [exec=0, time=0.0000s] [mem=781349 bytes]
        } elif direction == curses.KEY_DOWN {  # bb36 → bb38, bb37 [exec=1, time=0.0010s] [mem=781349 bytes]

            head[0] += 1;  # bb37 → bb42 [exec=0, time=0.0000s] [mem=781349 bytes]
        } elif direction == curses.KEY_RIGHT {  # bb38 → bb40, bb39 [exec=1, time=0.0011s] [mem=781349 bytes]

            head[1] += 1;  # bb39 → bb42 [exec=1, time=0.0192s] [mem=781349 bytes]
        } elif direction == curses.KEY_LEFT {  # bb40 → bb42, bb41 [exec=0, time=0.0000s] [mem=781349 bytes]

            head[1] -= 1;  # bb41 → bb42 [exec=0, time=0.0000s] [mem=781349 bytes]
        }
        
        if head[0] > (size - 1) {  # bb42 → bb44, bb43 [exec=1, time=0.0017s] [mem=781349 bytes]

            head[0] = 0;  # bb43 → bb50 [exec=0, time=0.0000s] [mem=781349 bytes]
        } elif head[0] < 0 {  # bb44 → bb46, bb45 [exec=1, time=0.0012s] [mem=781349 bytes]

            head[0] = (size - 1);  # bb45 → bb50 [exec=0, time=0.0000s] [mem=781349 bytes]
        } elif head[1] < 0 {  # bb46 → bb48, bb47 [exec=1, time=0.0012s] [mem=781349 bytes]

            head[1] = (size - 1);  # bb47 → bb50 [exec=0, time=0.0000s] [mem=781349 bytes]
        } elif head[1] > (size - 1) {  # bb48 → bb50, bb49 [exec=1, time=0.0017s] [mem=781349 bytes]

            head[1] = 0;  # bb49 → bb50 [exec=0, time=0.0000s] [mem=781349 bytes]
        }
        
        del (snake_coords[0]  # bb50 → bb52, bb51 [exec=1, time=0.0032s] [mem=781349 bytes]
        ,) ;    
        snake_coords.append(head) ;      # bb50 → bb52, bb51 [exec=1, time=0.0032s] [mem=781349 bytes]
        if head in snake_coords[:-1] {  # bb50 → bb52, bb51 [exec=1, time=0.0032s] [mem=781349 bytes]

            sys.exit() ;  # bb51 → bb52 [exec=0, time=0.0000s] [mem=781349 bytes]
        }
        
        entity_pos = [-1, -1];      # bb52 → bb53 [exec=1, time=0.0020s] [mem=781349 bytes]
        for i in range(size) {  # bb53 → bb61, bb54 [exec=1, time=0.0015s] [mem=781349 bytes] | bb52 → bb53 [exec=1, time=0.0020s] [mem=781349 bytes] | bb61 → bb62 [exec=0, time=0.0000s] [mem=781349 bytes]
            for j in range(size) {  # bb54 → bb57, bb56, bb55 [exec=11, time=0.0400s] [mem=781349 bytes] | bb53 → bb61, bb54 [exec=1, time=0.0015s] [mem=781349 bytes] | bb57 → bb58 [exec=0, time=0.0000s] [mem=781349 bytes]
                if field[i][j] == 3 {  # bb54 → bb57, bb56, bb55 [exec=11, time=0.0400s] [mem=781349 bytes] | bb55 → bb54 [exec=10, time=0.0021s] [mem=781349 bytes]

                    entity_pos = [i, j];      # bb56 → bb58 [exec=0, time=0.0000s] [mem=781349 bytes]
                    break ;  # bb56 → bb58 [exec=0, time=0.0000s] [mem=781349 bytes]
                }
            }

            
            if entity_pos != [-1, -1] {  # bb58 → bb60, bb59 [exec=0, time=0.0000s] [mem=781349 bytes] | bb59 → bb53 [exec=0, time=0.0000s] [mem=781349 bytes]
                break ;  # bb60 → bb62 [exec=0, time=0.0000s] [mem=781349 bytes]
            }
        }

        
        if entity_pos == head {  # bb62 → bb84, bb63 [exec=0, time=0.0000s] [mem=781349 bytes]

            curses.beep() ;      # bb63 → bb65, bb64 [exec=0, time=0.0000s] [mem=781349 bytes]
            a = snake_coords[0];      # bb63 → bb65, bb64 [exec=0, time=0.0000s] [mem=781349 bytes]
            b = snake_coords[1];      # bb63 → bb65, bb64 [exec=0, time=0.0000s] [mem=781349 bytes]
            tail = a[:];  # bb63 → bb65, bb64 [exec=0, time=0.0000s] [mem=781349 bytes]
            
            if a[0] < b[0] {  # bb63 → bb65, bb64 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[0] -= 1;  # bb64 → bb71 [exec=0, time=0.0000s] [mem=781349 bytes]
            } elif a[1] < b[1] {  # bb65 → bb67, bb66 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[1] -= 1;  # bb66 → bb71 [exec=0, time=0.0000s] [mem=781349 bytes]
            } elif a[0] > b[0] {  # bb67 → bb69, bb68 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[0] += 1;  # bb68 → bb71 [exec=0, time=0.0000s] [mem=781349 bytes]
            } elif a[1] > b[1] {  # bb69 → bb71, bb70 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[1] += 1;  # bb70 → bb71 [exec=0, time=0.0000s] [mem=781349 bytes]
            }
            
            if tail[0] > (size - 1) {  # bb71 → bb73, bb72 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[0] = 0;  # bb72 → bb79 [exec=0, time=0.0000s] [mem=781349 bytes]
            } elif tail[0] < 0 {  # bb73 → bb75, bb74 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[0] = (size - 1);  # bb74 → bb79 [exec=0, time=0.0000s] [mem=781349 bytes]
            } elif tail[1] < 0 {  # bb75 → bb77, bb76 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[1] = (size - 1);  # bb76 → bb79 [exec=0, time=0.0000s] [mem=781349 bytes]
            } elif tail[1] > (size - 1) {  # bb77 → bb79, bb78 [exec=0, time=0.0000s] [mem=781349 bytes]

                tail[1] = 0;  # bb78 → bb79 [exec=0, time=0.0000s] [mem=781349 bytes]
            }
            
            snake_coords.insert(0, tail) ;      # bb79 → bb84, bb80 [exec=0, time=0.0000s] [mem=781349 bytes]
            entity_added = False;      # bb79 → bb84, bb80 [exec=0, time=0.0000s] [mem=781349 bytes]
            while not entity_added {  # bb82 → bb84, bb83 [exec=0, time=0.0000s] [mem=781349 bytes] | bb83 → bb80 [exec=0, time=0.0000s] [mem=781349 bytes] | bb79 → bb84, bb80 [exec=0, time=0.0000s] [mem=781349 bytes]

                i = randint(0, (size - 1));  # bb80 → bb82, bb81 [exec=0, time=0.0000s] [mem=781349 bytes]
                
                j = randint(0, (size - 1));  # bb80 → bb82, bb81 [exec=0, time=0.0000s] [mem=781349 bytes]
                
                entity = [i, j];      # bb80 → bb82, bb81 [exec=0, time=0.0000s] [mem=781349 bytes]
                if entity not in snake_coords {  # bb80 → bb82, bb81 [exec=0, time=0.0000s] [mem=781349 bytes]

                    field[i][j] = 3;      # bb81 → bb82 [exec=0, time=0.0000s] [mem=781349 bytes]
                    entity_added = True;  # bb81 → bb82 [exec=0, time=0.0000s] [mem=781349 bytes]
                }
            }
        }
        
        field = [[0 for j in range(size)] for i in range(size)];      # bb85 → bb88, bb86 [exec=0, time=0.0000s] [mem=781349 bytes] | bb86 → bb87, bb86 [exec=0, time=0.0000s] [mem=781349 bytes] | bb84 → bb85 [exec=0, time=0.0000s] [mem=781349 bytes] | bb88 → bb89 [exec=0, time=0.0000s] [mem=781349 bytes] | bb87 → bb85 [exec=0, time=0.0000s] [mem=781349 bytes] | bb103 [exec=0, time=0.0000s] [mem=781349 bytes]
        for i in range(size) {  # bb88 → bb89 [exec=0, time=0.0000s] [mem=781349 bytes] | bb89 → bb96, bb90 [exec=0, time=0.0000s] [mem=781349 bytes] | bb96 → bb97 [exec=0, time=0.0000s] [mem=781349 bytes]
            for j in range(size) {  # bb89 → bb96, bb90 [exec=0, time=0.0000s] [mem=781349 bytes] | bb95 → bb89 [exec=0, time=0.0000s] [mem=781349 bytes] | bb90 → bb95, bb92, bb91 [exec=0, time=0.0000s] [mem=781349 bytes]
                if [i, j] == entity_pos and [i, j] != head {  # bb92 → bb94, bb93 [exec=0, time=0.0000s] [mem=781349 bytes] | bb93 → bb90 [exec=0, time=0.0000s] [mem=781349 bytes] | bb91 → bb90 [exec=0, time=0.0000s] [mem=781349 bytes] | bb90 → bb95, bb92, bb91 [exec=0, time=0.0000s] [mem=781349 bytes]

                    field[i][j] = 3;  # bb94 → bb90 [exec=0, time=0.0000s] [mem=781349 bytes]
                }
            }
        }
        
        for (i, j) in snake_coords {  # bb97 → bb98, bb97 [exec=0, time=0.0000s] [mem=781349 bytes] | bb96 → bb97 [exec=0, time=0.0000s] [mem=781349 bytes] | bb98 → bb99 [exec=0, time=0.0000s] [mem=781349 bytes]

            field[i][j] = 1;  # bb97 → bb98, bb97 [exec=0, time=0.0000s] [mem=781349 bytes]
        }
        
        head = snake_coords[-1];      # bb98 → bb99 [exec=0, time=0.0000s] [mem=781349 bytes]
        field[head[0]][head[1]] = 2;      # bb98 → bb99 [exec=0, time=0.0000s] [mem=781349 bytes]
        for i in range(size) {  # bb102 → bb20 [exec=0, time=0.0000s] [mem=781349 bytes] | bb99 → bb102, bb100 [exec=0, time=0.0000s] [mem=781349 bytes] | bb98 → bb99 [exec=0, time=0.0000s] [mem=781349 bytes]

            row = '';      # bb99 → bb102, bb100 [exec=0, time=0.0000s] [mem=781349 bytes]
            for j in range(size) {  # bb101 → bb99 [exec=0, time=0.0000s] [mem=781349 bytes] | bb99 → bb102, bb100 [exec=0, time=0.0000s] [mem=781349 bytes] | bb100 → bb101, bb100 [exec=0, time=0.0000s] [mem=781349 bytes]

                row += icons[field[i][j]];  # bb100 → bb101, bb100 [exec=0, time=0.0000s] [mem=781349 bytes]
            }
            
            screen.addstr(i, 0, row) ;  # bb101 → bb99 [exec=0, time=0.0000s] [mem=781349 bytes]
        }
        
        screen.refresh() ;      # bb102 → bb20 [exec=0, time=0.0000s] [mem=781349 bytes]
        time.sleep(0.4) ;  # bb102 → bb20 [exec=0, time=0.0000s] [mem=781349 bytes]
    }

    
}
