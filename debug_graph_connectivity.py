"""
Debug de la connectivité du graphe
"""

from data_parser import DataParser, RealisticGraphBuilder

def debug_graph_connectivity():
    """Vérifie si le graphe est bien connecté"""
    print("=== DEBUG CONNECTIVITÉ GRAPHE ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    # Test SOMAF
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        print(f"Graphe SOMAF:")
        print(f"  Intersections: {[i.name for i in tdg.intersections]}")
        print(f"  Source: {provider.location.name}")
        
        print(f"\nArêtes détaillées:")
        for i, edge in enumerate(tdg.edges):
            print(f"  {i+1}: {edge.origin.name} -> {edge.extremity.name} ({len(edge.chunks)} chunks)")
        
        # Vérifier la connectivité depuis X1 vers X9
        def find_path_bfs(start_name, end_name):
            """Recherche de chemin simple BFS"""
            from collections import deque
            
            # Construire la table d'adjacence
            adj = {}
            for intersection in tdg.intersections:
                adj[intersection.name] = []
            
            for edge in tdg.edges:
                adj[edge.origin.name].append(edge.extremity.name)
            
            print(f"\nTable d'adjacence:")
            for node, neighbors in adj.items():
                print(f"  {node}: {neighbors}")
            
            # BFS
            queue = deque([start_name])
            visited = {start_name}
            parent = {start_name: None}
            
            while queue:
                current = queue.popleft()
                
                if current == end_name:
                    # Reconstruire le chemin
                    path = []
                    node = end_name
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    return path[::-1]
                
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)
            
            return None
        
        # Test de connectivité
        path = find_path_bfs("X1", "X9")
        if path:
            print(f"\nChemin trouvé de X1 à X9: {' -> '.join(path)}")
        else:
            print(f"\nAucun chemin de X1 à X9!")
        
        # Vérifier toutes les connexions possibles
        print(f"\nToutes les connexions possibles depuis X1:")
        all_reachable = set()
        
        def dfs_reachable(start):
            visited = set()
            stack = [start]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                for edge in tdg.edges:
                    if edge.origin.name == current and edge.extremity.name not in visited:
                        stack.append(edge.extremity.name)
            
            return visited
        
        reachable = dfs_reachable("X1")
        print(f"Nœuds atteignables depuis X1: {sorted(reachable)}")
        
        if "X9" not in reachable:
            print("PROBLÈME: X9 n'est pas atteignable depuis X1!")
        else:
            print("OK: X9 est atteignable depuis X1")
    
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_graph_connectivity()