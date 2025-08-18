"""
Debug du MarkovStateManager
"""

import numpy as np
from data_parser import DataParser, RealisticGraphBuilder

def debug_state_manager():
    """Debug du gestionnaire d'états"""
    print("=== DEBUG STATE MANAGER ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        print(f"State manager initial:")
        print(f"  Cache: {len(builder.state_manager.edge_distributions)} entrées")
        
        # Test séquentiel de la fonction de poids
        print(f"\nTest séquentiel fonction de poids:")
        for i, edge in enumerate(tdg.edges):
            print(f"\n  Arête {i+1}: {edge.origin.name} -> {edge.extremity.name}")
            
            for time in [0.0, 6.0]:
                try:
                    # Appeler la fonction de poids
                    weight = tdg.weight_function(edge, float(time))
                    print(f"    Temps {time}h: {weight:.6f}h")
                    
                    # Vérifier l'état du cache
                    edge_key = f"{edge.origin.name}-{edge.extremity.name}"
                    if edge_key in builder.state_manager.edge_distributions:
                        dist = builder.state_manager.edge_distributions[edge_key]
                        print(f"    Distribution: {dist}")
                        print(f"    Somme dist: {np.sum(dist):.6f}")
                        
                        # Vérifier les valeurs problématiques
                        if np.any(np.isnan(dist)) or np.any(np.isinf(dist)):
                            print(f"    PROBLEME: Distribution corrompue!")
                        if np.sum(dist) < 0.5 or np.sum(dist) > 1.5:
                            print(f"    PROBLEME: Somme distribution invalide!")
                    
                    if np.isnan(weight) or np.isinf(weight):
                        print(f"    PROBLEME: Poids infini/NaN!")
                        break
                        
                except Exception as e:
                    print(f"    ERREUR: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        # Test reset du manager
        print(f"\n=== TEST RESET MANAGER ===")
        builder.state_manager = builder.__class__(suppliers, route_graphs).state_manager
        print("Manager reset")
        
        # Test avec un seul appel
        test_edge = tdg.edges[0]
        print(f"\nTest edge unique: {test_edge.origin.name} -> {test_edge.extremity.name}")
        
        try:
            weight1 = tdg.weight_function(test_edge, 0.0)
            print(f"Premier appel: {weight1:.6f}h")
            
            weight2 = tdg.weight_function(test_edge, 0.0)
            print(f"Deuxieme appel: {weight2:.6f}h")
            
            # Les deux devraient être similaires mais pas identiques à cause de la propagation
            
        except Exception as e:
            print(f"ERREUR: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_state_manager()