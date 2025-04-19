import:py from collections, deque ;  # bb2 → bb3 [exec=1, time=0.0015s] | bb0 → bb2, bb1 [exec=1, time=0.0012s] | bb1 → bb3 [exec=0, time=0.0000s]

class Node {  # bb3 → bb4 [exec=1, time=0.0000s]
    can init(name: Any, successors: Any=None) {

        self.name = name;    

        self.successors = successors or [];
    }
}

with entry {

    station1 = Node('Westminster');      # bb3 → bb4 [exec=1, time=0.0000s]
    station2 = Node('Waterloo', [station1]);      # bb3 → bb4 [exec=1, time=0.0000s]
    station3 = Node('Trafalgar Square', [station1, station2]);      # bb3 → bb4 [exec=1, time=0.0000s]
    station4 = Node('Canary Wharf', [station2, station3]);      # bb3 → bb4 [exec=1, time=0.0000s]
    station5 = Node('London Bridge', [station4, station3]);      # bb3 → bb4 [exec=1, time=0.0000s]
    station6 = Node('Tottenham Court Road', [station5, station4]);      # bb3 → bb4 [exec=1, time=0.0000s]
    startnode = station6;      # bb3 → bb4 [exec=1, time=0.0000s]
    goalnode = station1;      # bb3 → bb4 [exec=1, time=0.0000s]
    queue = deque();      # bb3 → bb4 [exec=1, time=0.0000s]
    queue.append(startnode) ;      # bb3 → bb4 [exec=1, time=0.0000s]
    nodesseen = <>set();      # bb3 → bb4 [exec=1, time=0.0000s]
    nodesseen.add(startnode) ;      # bb3 → bb4 [exec=1, time=0.0000s]
    found = False;      # bb3 → bb4 [exec=1, time=0.0000s]
    while True {  # bb3 → bb4 [exec=1, time=0.0000s] | bb6 → bb4 [exec=0, time=0.0000s]

        <>node = queue.popleft();      # bb4 → bb6, bb5 [exec=0, time=0.0000s]
        if <>node is goalnode {  # bb4 → bb6, bb5 [exec=0, time=0.0000s]

            found = True;      # bb5 → bb7 [exec=0, time=0.0000s]
            break ;  # bb5 → bb7 [exec=0, time=0.0000s]
        }

        
        queue.extend(  # bb6 → bb4 [exec=0, time=0.0000s]
            (n  for n in <>node.successors if n not in nodesseen)  # bb6 → bb4 [exec=0, time=0.0000s]
        ) ;    
        nodesseen.update(<>node.successors) ;  # bb6 → bb4 [exec=0, time=0.0000s]
    }
    
    print('Path found!' if found else 'Path not found!') ;  # bb7 → bb9, bb8 [exec=0, time=0.0000s] | bb9 [exec=0, time=0.0000s] | bb8 → bb9 [exec=0, time=0.0000s]
}
