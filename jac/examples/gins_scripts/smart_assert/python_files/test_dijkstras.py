from jaclang.runtimelib.gins.smart_assert import smart_assert
import heapq

def dijkstra(graph: dict[str, list[tuple[str,int]]], start: str) -> dict[str,int]:
    # BUG: ignores real weights, adds +1 for every edge
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            alt = d + 1      # should be d + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return dist

# A small weighted graph
graph1 = {
    'A': [('B', 2), ('C', 5)],
    'B': [('C', 1), ('D', 7)],
    'C': [('D', 3)],
    'D': []
}
expected1 = {'A': 0, 'B': 2, 'C': 3, 'D': 6}

# A graph with a disconnected node
graph2 = {
    'X': [('Y', 4)],
    'Y': [],
    'Z': []
}
expected2 = {'X': 0, 'Y': 4, 'Z': float('inf')}

smart_assert(dijkstra(graph1, 'A') == expected1)
smart_assert(dijkstra(graph2, 'X') == expected2)
