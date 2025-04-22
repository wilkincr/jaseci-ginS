import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;
import:py heapq ;

can dijkstra(graph: <>dict[(str, <>list[<>tuple[(str, int)]])], start: str) -> <>dict[str, int] {

    dist:dict = {<>node: float('inf')  for <>node in graph};    
    dist[start] = 0;    
    pq:list = [(0, start)];    
    
    while pq {
        d:int = 0;
        u:int = 0;
        (d, u) = heapq.heappop(pq);    
        if d > dist[u] {
            continue ;
        }

        v:str = "";
        w:int = 0;
        for (v, w) in graph[u] {

            alt:int = (d + 1);
            
            if alt < dist[v] {

                dist[v] = alt;    
                heapq.heappush(pq, (alt, v)) ;
            }
        }
    }
    
    return dist ;
}

with entry {

    graph1 = {
        'A': [('B', 2), ('C', 5)],
        'B': [('C', 1), ('D', 7)],
        'C': [('D', 3)],
        'D': []
    };
    
    expected1 = {'A': 0, 'B': 2, 'C': 3, 'D': 6};    
    graph2 = {'X': [('Y', 4)], 'Y': [], 'Z': []};
    
    expected2 = {'X': 0, 'Y': 4, 'Z': float('inf')};    
    smart_assert(dijkstra(graph1, 'A') == expected1) ;    
    smart_assert(dijkstra(graph2, 'X') == expected2) ;
}

