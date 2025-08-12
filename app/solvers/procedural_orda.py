from typing import Tuple
from app.models.tdg import TDG
from app.constants import INFINITY

class ProceduralOrdaSolver:

    @staticmethod
    def solve(tdg: TDG, v_s: int, v_e: int, T: Tuple[int, int]):
        # initialiser G_l
        g = dict()
        for v_l in tdg.V:
            g[v_l] = dict()
            for t in range(T[0], T[1]):
                g[v_l][t] = INFINITY

        # Initialiser H_k_l
        h = dict()
        for v_l in tdg.V:
            h[v_l] = dict()
        for v_k, v_l, _ in tdg.E:
            if v_k not in h:
                h[v_k] = dict()
            h[v_k][v_l] = dict()
            for t in range(T[0], T[1]):
                h[v_k][v_l][t] = INFINITY

        # initialiser g_s (coût 0 pour partir du source au temps t)
        for t in range(T[0], T[1]):
            g[v_s][t] = 0  # Corrigé: coût 0, pas t

        # on demarre le repeat
        iteration = 0
        max_iterations = len(tdg.V) * (T[1] - T[0])  # Protection contre boucles infinies
        
        while iteration < max_iterations:
            iteration += 1
            
            # mettre à jour les h_k_l(t)
            for v_k, v_l, chunks in tdg.E:
                for t in range(T[0], T[1]):
                    if g[v_k][t] < INFINITY:  # Seulement si le nœud est atteignable
                        arrival_time = g[v_k][t] + tdg.W((v_k, v_l, chunks), g[v_k][t])
                        h[v_k][v_l][t] = arrival_time

            # utilitaire pour la condition de rupture de la boucle
            can_stop = True

            # mettre à jour les g_l(t)
            for v_l in tdg.V:
                # Trouver tous les prédécesseurs de v_l
                n_l = []
                for v_k, v_i, _ in tdg.E:
                    if v_i == v_l:
                        n_l.append(v_k)
                
                for t in range(T[0], T[1]):
                    if n_l:  # Seulement si v_l a des prédécesseurs
                        new_value = min(h[v_k][v_l][t] for v_k in n_l)
                        if abs(g[v_l][t] - new_value) > 1e-10:  # Corrigé: comparaison numérique
                            can_stop = False
                        g[v_l][t] = new_value

            if can_stop:
                break

        # Récupérer le meilleur instant de départ
        t_star = T[0]
        for t in range(T[0], T[1]):
            if g[v_e][t] < g[v_e][t_star]:
                t_star = t

        # Construire le chemin optimal (reconstruction du chemin)
        current_node = v_e
        current_time = t_star
        p_star = []
        
        while current_node != v_s:
            # Trouver l'arête qui a donné le coût optimal
            found = False
            for v_i, v_j, chunks in tdg.E:
                if v_j == current_node:
                    # Calculer le temps de départ nécessaire depuis v_i
                    weight = tdg.W((v_i, v_j, chunks), g[v_i][current_time])
                    expected_arrival = g[v_i][current_time] + weight
                    
                    if abs(expected_arrival - g[current_node][current_time]) < 1e-10:
                        p_star.insert(0, (v_i, v_j))  # Insérer au début
                        current_node = v_i
                        # Trouver le temps optimal pour arriver à v_i
                        # (simplification: on garde le même temps, mais dans un vrai cas
                        # il faudrait calculer le temps d'arrivée à v_i)
                        found = True
                        break
            
            if not found:
                # Impossible de reconstruire le chemin
                break

        # return the best starting time and the best path
        return t_star, p_star
