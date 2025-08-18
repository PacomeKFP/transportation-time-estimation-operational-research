"""
Test manuel de calcul de chemin sans utiliser le solveur TwoStepLTT
"""

import numpy as np
from data_parser import DataParser, RealisticGraphBuilder

def test_manual_path():
    """Test manuel du calcul de chemin"""
    print("=== TEST MANUEL CHEMIN ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        print(f"Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
        print(f"Source: {provider.location.name}")
        
        # Calculer manuellement le coût du chemin X1 -> X2 -> X5 -> X8 -> X9
        path_nodes = ["X1", "X2", "X5", "X8", "X9"]
        print(f"\nChemin manuel: {' -> '.join(path_nodes)}")
        
        total_cost = 0.0
        current_time = 0.0  # Départ à 0h
        
        for i in range(len(path_nodes) - 1):
            start_node = path_nodes[i]
            end_node = path_nodes[i + 1]
            
            # Trouver l'arête correspondante
            edge_found = None
            for edge in tdg.edges:
                if edge.origin.name == start_node and edge.extremity.name == end_node:
                    edge_found = edge
                    break
            
            if edge_found:
                travel_time = tdg.weight_function(edge_found, current_time)
                total_cost += travel_time
                current_time += travel_time
                
                print(f"  {start_node} -> {end_node}: {travel_time:.6f}h (départ: {current_time - travel_time:.2f}h)")
            else:
                print(f"  ERREUR: Arête {start_node} -> {end_node} non trouvée!")
                return
        
        print(f"\nCoût total manuel: {total_cost:.6f}h")
        print(f"Temps d'arrivée: {current_time:.6f}h")
        
        # Comparer avec les valeurs individuelles
        print(f"\n=== VERIFICATION INDIVIDUELLE ===")
        for edge in tdg.edges:
            for test_time in [0.0, 1.0, 2.0]:
                weight = tdg.weight_function(edge, test_time)
                print(f"  {edge.origin.name}->{edge.extremity.name} @ {test_time}h: {weight:.6f}h")
        
        # Test simple avec fonction de poids constante
        print(f"\n=== TEST FONCTION SIMPLE ===")
        
        def simple_weight(edge, departure_time):
            if departure_time == float('inf') or departure_time == float('-inf') or departure_time != departure_time:
                return float('inf')
            total_length = sum(chunk.length for chunk in edge.chunks)
            return float(total_length * 0.02)  # 20 min/km = 0.02h/km
        
        # Remplacer temporairement
        original_weight = tdg.weight_function
        tdg.weight_function = simple_weight
        
        # Test du solveur avec fonction simple
        from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed
        
        destination = None
        for intersection in tdg.intersections:
            if intersection.name == "X9":
                destination = intersection
                break
        
        print(f"Test solveur avec fonction simple:")
        try:
            result = TwoStepLTTFixed.solve(tdg, provider.location, destination, (0, 5))
            if result.path is not None:
                print(f"  OK: {result.total_cost:.6f}h, départ: {result.t_star:.2f}h")
                path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in result.path.edges]
                print(f"  Chemin: {' -> '.join(path_edges)}")
            else:
                print(f"  NOK: {result.total_cost}")
        except Exception as e:
            print(f"  ERREUR: {e}")
        
        # Remettre la fonction originale
        tdg.weight_function = original_weight
        
        print(f"\nTest solveur avec fonction Markov:")
        try:
            result2 = TwoStepLTTFixed.solve(tdg, provider.location, destination, (0, 5))
            if result2.path is not None:
                print(f"  OK: {result2.total_cost:.6f}h, départ: {result2.t_star:.2f}h")
            else:
                print(f"  NOK: {result2.total_cost}")
        except Exception as e:
            print(f"  ERREUR: {e}")
    
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_manual_path()