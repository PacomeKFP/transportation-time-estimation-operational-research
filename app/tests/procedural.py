from typing import Set, Tuple, List, Callable
from app.models.tdg import TDG
from app.solvers.procedural_orda import ProceduralOrdaSolver


def exemple_complexe():
    """
    Graphe représentant un réseau urbain avec embouteillages :
    
    Nœuds: 0 (Maison) -> 1 (Centre-ville) -> 4 (Bureau)
           0 (Maison) -> 2 (Rocade) -> 3 (Périphérie) -> 4 (Bureau)
           
    Temps simulé : heures de la journée (0-23h)
    """
    
    def weight_function(edge, departure_time):
        """
        Fonction de poids simulant les embouteillages :
        - Route directe (0->1->4) : rapide sauf aux heures de pointe
        - Route rocade (0->2->3->4) : plus longue mais stable
        """
        v_from, v_to, metadata = edge
        hour = int(departure_time) % 24  # Convertir en heure de la journée
        
        if (v_from, v_to) == (0, 1):  # Maison -> Centre-ville
            # Embouteillages 7-9h et 17-19h
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 8.0  # Très lent aux heures de pointe
            else:
                return 2.0  # Rapide sinon
                
        elif (v_from, v_to) == (1, 4):  # Centre-ville -> Bureau
            # Embouteillages 7-9h et 17-19h
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 10.0  # Très lent aux heures de pointe
            else:
                return 3.0  # Rapide sinon
                
        elif (v_from, v_to) == (0, 2):  # Maison -> Rocade
            return 4.0  # Coût constant
            
        elif (v_from, v_to) == (2, 3):  # Rocade -> Périphérie
            return 3.0  # Coût constant
            
        elif (v_from, v_to) == (3, 4):  # Périphérie -> Bureau
            return 2.0  # Coût constant
            
        else:
            return float('inf')  # Arête inexistante
    
    # Créer le graphe : 5 nœuds avec 2 chemins alternatifs
    edges = {
        (0, 1, ["route_directe"]),     # Maison -> Centre-ville
        (1, 4, ["route_directe"]),     # Centre-ville -> Bureau
        (0, 2, ["rocade"]),            # Maison -> Rocade
        (2, 3, ["rocade"]),            # Rocade -> Périphérie  
        (3, 4, ["rocade"])             # Périphérie -> Bureau
    }
    
    tdg = TDG(5, edges, weight_function)
    
    print("=== ANALYSE DU RÉSEAU DE TRANSPORT ===")
    print("Nœuds: 0=Maison, 1=Centre-ville, 2=Rocade, 3=Périphérie, 4=Bureau")
    print("Routes:")
    print("  - Directe: 0 -> 1 -> 4 (rapide hors embouteillages)")
    print("  - Rocade:  0 -> 2 -> 3 -> 4 (plus longue mais stable)")
    print()
    
    # Tester différents créneaux horaires
    test_periods = [
        (6, 10, "Matin (6h-9h) - avec embouteillages"),
        (10, 14, "Milieu de journée (10h-13h) - fluide"), 
        (17, 21, "Soir (17h-20h) - avec embouteillages"),
        (22, 26, "Nuit (22h-1h) - fluide")  # 26 = 2h du matin
    ]
    
    for start_time, end_time, description in test_periods:
        print(f"--- {description} ---")
        t_star, path, _ = ProceduralOrdaSolver.solve(tdg, 0, 4, (start_time, end_time))
        
        # Calculer le coût total du chemin
        total_cost = 0
        current_time = t_star
        for v_from, v_to in path:
            edge = (v_from, v_to, [])
            weight = tdg.W(edge, current_time)
            total_cost += weight
            current_time += weight
            
        print(f"Meilleur départ: {t_star}h")
        print(f"Chemin optimal: {path}")
        print(f"Coût total: {total_cost:.1f}")
        
        # Interpréter le chemin
        if len(path) == 2 and path[0] == (0, 1):
            print("→ Route DIRECTE choisie (via centre-ville)")
        elif len(path) == 3 and path[0] == (0, 2):
            print("→ Route ROCADE choisie (évite centre-ville)")
        print()
    
    print("=== COMPARAISON DÉTAILLÉE À 8H (HEURE DE POINTE) ===")
    
    # Analyser les coûts de chaque route à 8h
    hour = 8
    print(f"Coûts à {hour}h:")
    
    # Route directe
    cost_direct = weight_function((0, 1, []), hour) + weight_function((1, 4, []), hour)
    print(f"Route directe (0->1->4): {cost_direct:.1f}")
    
    # Route rocade
    cost_rocade = (weight_function((0, 2, []), hour) + 
                   weight_function((2, 3, []), hour) + 
                   weight_function((3, 4, []), hour))
    print(f"Route rocade (0->2->3->4): {cost_rocade:.1f}")
    
    if cost_direct < cost_rocade:
        print("→ Route directe plus rapide malgré les embouteillages")
    else:
        print("→ Route rocade plus rapide grâce à l'évitement des embouteillages")


def exemple_simple():
    """Exemple simple pour vérification"""
    def weight_function(edge, departure_time):
        return 1.0  # Poids constant
    
    # Créer un graphe simple : 0 -> 1 -> 2
    edges = {
        (0, 1, []),
        (1, 2, [])
    }
    
    tdg = TDG(3, edges, weight_function)
    
    # Chercher le plus court chemin de 0 à 2 entre les temps 0 et 5
    t_star, path, _ = ProceduralOrdaSolver.solve(tdg, 0, 2, (0, 5))
    
    print("=== EXEMPLE SIMPLE ===")
    print(f"Meilleur temps de départ: {t_star}")
    print(f"Chemin optimal: {path}")
    print()
