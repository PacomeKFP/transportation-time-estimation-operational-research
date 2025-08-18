"""Test direct du solveur avec données réelles"""

from data_parser import DataParser, RealisticGraphBuilder
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed

def test_solver_direct():
    """Test direct avec SOMAF"""
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    # Test avec SOMAF
    tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
    
    print("=== TEST SOLVEUR DIRECT ===")
    print(f"Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} aretes")
    print(f"Source: {provider.location.name}")
    
    # Trouver destination
    destination = None
    max_num = -1
    for intersection in tdg.intersections:
        if intersection.name.startswith('X'):
            try:
                num = int(intersection.name[1:])
                if num > max_num:
                    max_num = num
                    destination = intersection
            except ValueError:
                continue
    
    print(f"Destination: {destination.name}")
    
    # Test de la fonction de poids
    test_edge = tdg.edges[0]
    for time in [0, 6, 12, 18]:
        weight = tdg.weight_function(test_edge, float(time))
        print(f"Poids à {time}h: {weight}")
    
    # Test avec différentes fenêtres temporelles
    for window in [(0, 3), (0, 5), (0, 10), (12, 20)]:
        print(f"\n--- Test avec fenetre {window} ---")
        try:
            result = TwoStepLTTFixed.solve(tdg, provider.location, destination, window)
            
            if result.path is not None:
                print(f"  OK - Temps: {result.t_star}, Cout: {result.total_cost}")
                path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in result.path.edges]
                print(f"  Chemin: {' -> '.join(path_edges)}")
            else:
                print(f"  NOK - Cout: {result.total_cost}")
                
        except Exception as e:
            print(f"  ERREUR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_solver_direct()