class Node:
    def __init__(self, name, successors=None):
        self.name = name
        self.successors = successors or []

# Build graph with cycle
node1 = Node("1")
node2 = Node("2")
node3 = Node("3")
node4 = Node("4", [node1])
node5 = Node("5", [node2])
node6 = Node("6", [node5, node4, node3])
node2.successors = [node6]  # Creates cycle

startnode = node6
goalnode = node1

stack = [startnode]
visited = set()
found = False

while stack:
    node = stack.pop()
    if node in visited:
        continue
    if node is goalnode:
        found = True
        break
    visited.add(node)
    stack.extend(reversed(node.successors))

print("Path found!" if found else "Path not found!")
