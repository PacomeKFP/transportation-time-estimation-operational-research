---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[2], line 5
      2 from app.solvers.two_step_ltt_2008_solver import TwoStepLTT2008Solver
      4 # Résoudre le problème
----> 5 result = TwoStepLTT2008Solver.solve(
      6     tdg=graph,
      7     source=i2,
      8     destination=i10,
      9     time_window=(0, 24)  # 24 heures
     10 )
     12 print(f"Meilleur départ : {result.t_star}h")
     13 print(f"Coût total : {result.total_cost:.2f}")

File c:\workspace\clients\transportation-time-estimation-operational-research\app\solvers\two_step_ltt_2008_solver.py:287, in TwoStepLTT2008Solver.solve(tdg, source, destination, time_window)
    284     print("Attention: Le graphe n'est pas FIFO. L'algorithme peut ne pas être optimal.")
    286 # Étape 1: Time Refinement (ligne 1)
--> 287 g_functions = TwoStepLTT2008Solver.timeRefinement(tdg, source, destination, time_window)
    289 # Vérification de l'atteignabilité (ligne 2)
    290 g_e = g_functions[destination]

File c:\workspace\clients\transportation-time-estimation-operational-research\app\solvers\two_step_ltt_2008_solver.py:100, in TwoStepLTT2008Solver.timeRefinement(tdg, source, destination, time_interval)
     98 for node in tdg.intersections:
     99     priority = g[node](tau[node])
--> 100     heapq.heappush(Q, (priority, tau[node], node, g[node]))
    102 # Boucle principale (lignes 5-19)
    103 while len(Q) >= 2:
    104     # Dequeue le nœud avec la plus petite arrivée (ligne 6)

TypeError: '<' not supported between instances of 'Intersection' and 'Intersection'