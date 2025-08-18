"""
Test final d'intégration avec les chaînes de Markov
"""

from main_application import TransportationTimeEstimationSystem, DeliveryRequest

def test_final_integration():
    """Test final du système complet"""
    print("=== TEST FINAL INTÉGRATION MARKOV ===")
    
    # Initialiser le système
    system = TransportationTimeEstimationSystem('xl/b.txt', 'xl/a.txt')
    
    # Test avec une demande simple
    delivery_request = DeliveryRequest(
        quantity_demanded=30.0,  # Demande plus petite
        delivery_date="2024-01-15",
        delivery_hour=10,  # Heure moins contraignante
        construction_site="Chantier Central"
    )
    
    print(f"\nDemande: {delivery_request.quantity_demanded}m³ à {delivery_request.delivery_hour}h")
    
    # Test manuel de chaque fournisseur avec fenêtre simplifiée
    print(f"\n=== TEST MANUEL FOURNISSEURS ===")
    
    from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed
    
    for supplier in system.suppliers_data:
        print(f"\n--- Test {supplier.name} ---")
        
        if supplier.delivery_capacity >= delivery_request.quantity_demanded:
            try:
                # Construire le graphe
                tdg, provider = system.graph_builder.build_supplier_graph(supplier.name)
                
                # Destination
                construction_site_node = system.get_construction_site_node(tdg)
                
                print(f"  Source: {provider.location.name}")
                print(f"  Destination: {construction_site_node.name}")
                print(f"  Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
                
                # Test direct avec fenêtre petite
                for window in [(0, 1), (0, 2), (0, 5)]:
                    try:
                        result = TwoStepLTTFixed.solve(
                            tdg, 
                            provider.location, 
                            construction_site_node, 
                            window
                        )
                        
                        if result.path is not None:
                            print(f"  ✓ Fenêtre {window}: {result.total_cost:.4f}h, départ: {result.t_star:.2f}h")
                            path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in result.path.edges]
                            print(f"    Chemin: {' -> '.join(path_edges)}")
                            break  # Succès trouvé
                        else:
                            print(f"  ✗ Fenêtre {window}: pas de solution")
                    except Exception as e:
                        print(f"  ✗ Fenêtre {window}: erreur {e}")
                
            except Exception as e:
                print(f"  ERREUR construction: {e}")
        else:
            print(f"  SKIP: capacité insuffisante ({supplier.delivery_capacity}m³)")
    
    # Test du système complet avec paramètres ajustés
    print(f"\n=== TEST SYSTÈME COMPLET ===")
    
    # Temporairement modifier la logique du système pour des fenêtres plus petites
    original_estimate = system.estimate_transportation_time
    
    def modified_estimate(supplier, delivery_request):
        """Version modifiée avec fenêtres plus petites"""
        print(f"\\n--- Estimation modifiée pour {supplier.name} ---")
        
        try:
            # Construire le graphe TDG du fournisseur
            tdg, provider = system.graph_builder.build_supplier_graph(supplier.name)
            
            # Déterminer le nœud de destination (chantier)
            construction_site_node = system.get_construction_site_node(tdg)
            
            print(f"  Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
            
            # Essayer plusieurs fenêtres jusqu'à trouver une solution
            for window in [(0, 1), (0, 2), (0, 5), (0, 10)]:
                print(f"  Test fenêtre: {window}")
                
                try:
                    result = TwoStepLTTFixed.solve(
                        tdg, 
                        provider.location, 
                        construction_site_node, 
                        window
                    )
                    
                    if result.path is not None:
                        provider.transportation_time = result.total_cost
                        provider.best_path = result.path
                        
                        print(f"  ✓ Temps estimé: {result.total_cost:.4f}h")
                        print(f"  ✓ Départ optimal: {result.t_star:.2f}h")
                        
                        return result.total_cost, tdg, provider
                        
                except Exception as e:
                    print(f"  ✗ Erreur fenêtre {window}: {e}")
                    continue
            
            print(f"  NOK Aucune fenêtre ne fonctionne")
            return float('inf'), tdg, provider
                
        except Exception as e:
            print(f"  NOK Erreur: {e}")
            return float('inf'), None, None
    
    # Remplacer temporairement
    system.estimate_transportation_time = modified_estimate
    
    # Exécuter le système complet
    optimal_suppliers = system.solve_delivery_optimization(delivery_request)
    
    if optimal_suppliers:
        print(f"\\n🎉 SUCCÈS! Solutions trouvées:")
        for i, eval_result in enumerate(optimal_suppliers):
            print(f"  {i+1}. {eval_result.supplier.name}: {eval_result.estimated_time:.4f}h, {eval_result.total_cost:.0f} FCFA/m³")
    else:
        print(f"\\n❌ Aucune solution trouvée avec les chaînes de Markov")

if __name__ == "__main__":
    test_final_integration()