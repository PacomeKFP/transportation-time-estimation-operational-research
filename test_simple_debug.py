"""
Test simple avec logs détaillés
"""

from data_parser import DataParser, RealisticGraphBuilder
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed

def test_simple_debug():
    """Test simple avec SOMAF"""
    print("=== TEST SIMPLE AVEC LOGS ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        # Destination
        destination = None
        for intersection in tdg.intersections:
            if intersection.name == "X9":
                destination = intersection
                break
        
        print(f"Test: {provider.location.name} -> {destination.name}")
        print(f"Arêtes: {[(e.origin.name, e.extremity.name) for e in tdg.edges]}")
        
        # Test avec fenêtre très petite
        time_window = (0, 2)
        print(f"Fenêtre: {time_window}")
        
        result = TwoStepLTTFixed.solve(tdg, provider.location, destination, time_window)
        
        print(f"Résultat final: path={result.path is not None}, cost={result.total_cost}")
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_debug()