import curses
import time
import sys
from random import randint

if __name__ == '__main__':
    # Wrap everything in the curses wrapper
    # This will initialize the terminal for curses use and restore it when done
    try:
        # Initialize the screen
        screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        screen.keypad(True)
        screen.timeout(0)
        
        # Field size
        size = 10
        
        # Icons for different elements
        icons = {
            0: ' . ',  # Empty
            1: ' * ',  # Snake body
            2: ' # ',  # Snake head
            3: ' & '   # Food entity
        }
        
        # Initial field generation
        field = [[0 for j in range(size)] for i in range(size)]
        
        # Initial snake position and direction
        snake_coords = [[0, 0], [0, 1], [0, 2], [0, 3]]
        direction = curses.KEY_RIGHT
        
        # Add initial food entity
        i = randint(0, size-1)
        j = randint(0, size-1)
        entity = [i, j]
        
        while entity in snake_coords:
            i = randint(0, size-1)
            j = randint(0, size-1)
            entity = [i, j]
            
        field[i][j] = 3
        
        # Main game loop
        while True:
            # Get last pressed key
            ch = screen.getch()
            if ch != -1:
                # Validate direction change
                invalid_direction = (
                    (ch == curses.KEY_LEFT and direction == curses.KEY_RIGHT) or
                    (ch == curses.KEY_RIGHT and direction == curses.KEY_LEFT) or
                    (ch == curses.KEY_UP and direction == curses.KEY_DOWN) or
                    (ch == curses.KEY_DOWN and direction == curses.KEY_UP)
                )
                if not invalid_direction:
                    direction = ch
            
            # Move snake
            head = snake_coords[-1][:]
            
            # Calculate new head position based on direction
            if direction == curses.KEY_UP:
                head[0] -= 1
            elif direction == curses.KEY_DOWN:
                head[0] += 1
            elif direction == curses.KEY_RIGHT:
                head[1] += 1
            elif direction == curses.KEY_LEFT:
                head[1] -= 1
            
            # Check field limits
            if head[0] > size-1:
                head[0] = 0
            elif head[0] < 0:
                head[0] = size-1
            elif head[1] < 0:
                head[1] = size-1
            elif head[1] > size-1:
                head[1] = 0
            
            # Remove tail and add new head
            del(snake_coords[0])
            snake_coords.append(head)
            
            # Check if snake is alive
            if head in snake_coords[:-1]:
                sys.exit()
            
            # Check if snake eats food
            entity_pos = [-1, -1]
            for i in range(size):
                for j in range(size):
                    if field[i][j] == 3:
                        entity_pos = [i, j]
                        break
                if entity_pos != [-1, -1]:
                    break
            
            if entity_pos == head:
                curses.beep()
                
                # Level up - add segment to snake
                a = snake_coords[0]
                b = snake_coords[1]
                tail = a[:]
                
                if a[0] < b[0]:
                    tail[0] -= 1
                elif a[1] < b[1]:
                    tail[1] -= 1
                elif a[0] > b[0]:
                    tail[0] += 1
                elif a[1] > b[1]:
                    tail[1] += 1
                
                # Check field limits for new tail
                if tail[0] > size-1:
                    tail[0] = 0
                elif tail[0] < 0:
                    tail[0] = size-1
                elif tail[1] < 0:
                    tail[1] = size-1
                elif tail[1] > size-1:
                    tail[1] = 0
                
                snake_coords.insert(0, tail)
                
                # Add new food entity
                entity_added = False
                while not entity_added:
                    i = randint(0, size-1)
                    j = randint(0, size-1)
                    entity = [i, j]
                    
                    if entity not in snake_coords:
                        field[i][j] = 3
                        entity_added = True
            
            # Clear field for rendering
            field = [[0 for j in range(size)] for i in range(size)]
            
            # Add food entity back to field
            for i in range(size):
                for j in range(size):
                    if [i, j] == entity_pos and [i, j] != head:
                        field[i][j] = 3
            
            # Render snake on the field
            for i, j in snake_coords:
                field[i][j] = 1
            
            # Mark head
            head = snake_coords[-1]
            field[head[0]][head[1]] = 2
            
            # Render field
            for i in range(size):
                row = ''
                for j in range(size):
                    row += icons[field[i][j]]
                screen.addstr(i, 0, row)
            
            screen.refresh()
            time.sleep(0.4)
            
    except Exception as e:
        pass
    finally:
        # Cleanup when done
        curses.nocbreak()
        screen.keypad(False)
        curses.echo()
        curses.endwin()