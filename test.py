from collections import defaultdict, deque

def validPath(n, edges, source, destination):
    # Step 1: Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    print(f"Graph: {dict(graph)}")
    
    # Step 2: BFS
    queue = deque([source])
    visited = set()
    visited.add(source)
    
    print(f"\nStart BFS from node {source}, looking for node {destination}")
    iteration = 0
    
    while queue:
        iteration += 1
        node = queue.popleft()
        print(f"\nIteration {iteration}: popped node {node}")
        
        # Step 3: found destination?
        if node == destination:
            print(f"Found destination {destination}! return True")
            return True
        
        # Step 4: add unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                print(f"  Added neighbor {neighbor} to queue")
        
        print(f"  Queue now: {list(queue)}")
        print(f"  Visited: {visited}")
    
    print(f"\nQueue empty, destination not found → return False")
    return False

# Test 1: no path exists
print("="*50)
print("TEST 1: source=0, destination=5")
print("="*50)
n = 6
edges = [[0,1],[0,2],[3,4],[4,5]]
print(validPath(n, edges, 0, 5))

# Test 2: path exists
print("\n" + "="*50)
print("TEST 2: source=3, destination=5")
print("="*50)
print(validPath(n, edges, 3, 5))

# Test 3: source == destination
print("\n" + "="*50)
print("TEST 3: source=0, destination=0")
print("="*50)
print(validPath(n, edges, 0, 0))