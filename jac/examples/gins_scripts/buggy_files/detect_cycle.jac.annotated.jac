class Node {  # bb0 → bb1 [exec=1, time=0.0000s]
    can init(val: Any, successor: Any=None) {

        self.val = val;    
        self.successor = successor;
    }
}

with entry {

    node1 = Node(1);      # bb0 → bb1 [exec=1, time=0.0000s]

    node2 = Node(2, node1);      # bb0 → bb1 [exec=1, time=0.0000s]

    node3 = Node(3, node2);      # bb0 → bb1 [exec=1, time=0.0000s]

    node4 = Node(4, node3);      # bb0 → bb1 [exec=1, time=0.0000s]

    node5 = Node(5, node4);      # bb0 → bb1 [exec=1, time=0.0000s]

    node1.successor = None;      # bb0 → bb1 [exec=1, time=0.0000s]

    hare = node5;      # bb0 → bb1 [exec=1, time=0.0000s]

    tortoise = node5;      # bb0 → bb1 [exec=1, time=0.0000s]
    while True {  # bb0 → bb1 [exec=1, time=0.0000s] | bb3 → bb1 [exec=0, time=0.0000s]
        if hare.successor is None {  # bb1 → bb3, bb2 [exec=0, time=0.0000s]

            print('No cycle') ;      # bb1 → bb3, bb2 [exec=0, time=0.0000s]
            break ;  # bb1 → bb3, bb2 [exec=0, time=0.0000s]
        }

        

        tortoise = tortoise.successor;      # bb1 → bb3, bb2 [exec=0, time=0.0000s]

        hare = hare.successor.successor;      # bb1 → bb3, bb2 [exec=0, time=0.0000s]
        if hare is tortoise {  # bb1 → bb3, bb2 [exec=0, time=0.0000s]

            print('Cycle detected') ;      # bb2 → bb3 [exec=0, time=0.0000s]
            break ;  # bb2 → bb3 [exec=0, time=0.0000s]
        }
    }
}
