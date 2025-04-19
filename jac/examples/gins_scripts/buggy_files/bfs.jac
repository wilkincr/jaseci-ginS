import:py from collections, deque ;

class Node {
    can init(name: Any, successors: Any=None) {

        self.name = name;    

        self.successors = successors or [];
    }
}

with entry {

    station1 = Node('Westminster');    
    station2 = Node('Waterloo', [station1]);    
    station3 = Node('Trafalgar Square', [station1, station2]);    
    station4 = Node('Canary Wharf', [station2, station3]);    
    station5 = Node('London Bridge', [station4, station3]);    
    station6 = Node('Tottenham Court Road', [station5, station4]);    
    startnode = station6;    
    goalnode = station1;    
    queue = deque();    
    queue.append(startnode) ;    
    nodesseen = <>set();    
    nodesseen.add(startnode) ;    
    found = False;    
    while True {

        <>node = queue.popleft();    
        if <>node is goalnode {

            found = True;    
            break ;
        }

        
        queue.extend(
            (n  for n in <>node.successors if n not in nodesseen)
        ) ;    
        nodesseen.update(<>node.successors) ;
    }
    
    print('Path found!' if found else 'Path not found!') ;
}

