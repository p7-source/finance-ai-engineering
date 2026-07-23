from collections import deque

class Node:
    def __init__(self, val=0, neighbours=None):
        self.val = val
        self.neighbours = neighbours if neighbours else []

def clone(node):
    if not node:
        print("Empty graph → return None")
        return None

    queue = deque([node])
    clones = {node.val: Node(node.val)}

    print(f"\n--- START ---")
    print(f"Initial node: {node.val}")
    print(f"clones dict: {{{node.val}: Node({node.val})'}} ")
    print(f"queue: [{node.val}]")

    iteration = 0

    while queue:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"{'='*50}")

        curr = queue.popleft()
        curr_clone = clones[curr.val]

        print(f"curr         = node{curr.val} (original)")
        print(f"curr_clone   = Node({curr.val})' (its copy)")
        print(f"curr.neighbours = {[n.val for n in curr.neighbours]}")

        for neighbour in curr.neighbours:
            print(f"\n  → Processing neighbour: node{neighbour.val}")

            if neighbour.val not in clones:
                print(f"    node{neighbour.val} NOT in clones → create Node({neighbour.val})'")
                clones[neighbour.val] = Node(neighbour.val)
                queue.append(neighbour)
                print(f"    Added node{neighbour.val} to queue")
            else:
                print(f"    node{neighbour.val} ALREADY in clones → skip creating, just connect")

            print(f"    curr_clone.neighbours.append(clones[{neighbour.val}])")
            print(f"    = Node({curr.val})'.neighbours.append(Node({neighbour.val})')")
            curr_clone.neighbours.append(clones[neighbour.val])
            print(f"    Node({curr.val})' neighbours now: {[n.val for n in curr_clone.neighbours]}")

        print(f"\n  clones dict keys: {list(clones.keys())}")
        print(f"  queue remaining: {[n.val for n in queue]}")

    print(f"\n{'='*50}")
    print(f"Queue empty — cloning complete!")
    print(f"Returning clones[{node.val}] = Node({node.val})'")

    return clones[node.val]

# ─────────────────────────────────────────
# Build the graph from adjList
# adjList = [[2,4],[1,3],[2,4],[1,3]]
# Node 1 neighbours: [2,4]
# Node 2 neighbours: [1,3]
# Node 3 neighbours: [2,4]
# Node 4 neighbours: [1,3]
# ─────────────────────────────────────────
print("Building original graph...")
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)

node1.neighbours = [node2, node4]
node2.neighbours = [node1, node3]
node3.neighbours = [node2, node4]
node4.neighbours = [node1, node3]

print(f"node1 neighbours: {[n.val for n in node1.neighbours]}")
print(f"node2 neighbours: {[n.val for n in node2.neighbours]}")
print(f"node3 neighbours: {[n.val for n in node3.neighbours]}")
print(f"node4 neighbours: {[n.val for n in node4.neighbours]}")

# Clone the graph
result = clone(node1)

# Verify the clone
print(f"\n{'='*50}")
print("VERIFYING CLONE:")
print(f"{'='*50}")
print(f"Node(1)' neighbours: {[n.val for n in result.neighbours]}")
print(f"Node(2)' neighbours: {[n.val for n in result.neighbours[0].neighbours]}")
print(f"Node(3)' neighbours: {[n.val for n in result.neighbours[0].neighbours[1].neighbours]}")
print(f"Node(4)' neighbours: {[n.val for n in result.neighbours[1].neighbours]}")

print(f"\nOriginal node1 id: {id(node1)}")
print(f"Cloned  node1 id: {id(result)}")
print(f"Are they different objects? {id(node1) != id(result)}")