class Node {
    can init(val: Any, successor: Any=None) {

        self.val = val;    
        self.successor = successor;
    }
}

with entry {

    node1 = Node(1);    

    node2 = Node(2, node1);    

    node3 = Node(3, node2);    

    node4 = Node(4, node3);    

    node5 = Node(5, node4);    

    node1.successor = None;    

    hare = node5;    

    tortoise = node5;    
    while True {
        if hare.successor is None {

            print('No cycle') ;    
            break ;
        }

        

        tortoise = tortoise.successor;    

        hare = hare.successor.successor;    
        if hare is tortoise {

            print('Cycle detected') ;    
            break ;
        }
    }
}

