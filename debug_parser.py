"""Debug du parser pour voir pourquoi il ne lit qu'un tronçon par arc"""

def debug_parse():
    with open('xl/a.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_supplier = None
    current_arc = None
    graphs = {}
    
    for i, line in enumerate(lines[:20]):  # Premiers 20 lignes pour debug
        line_stripped = line.strip()
        print(f"Ligne {i:2}: {repr(line_stripped)}")
        
        if line_stripped.startswith('FOURNISSEUR'):
            current_supplier = line_stripped.replace('FOURNISSEUR', '').strip()
            graphs[current_supplier] = []
            current_arc = None
            print(f"  --> Nouveau fournisseur: {current_supplier}")
            continue
            
        if line_stripped.startswith('Arcs'):
            print("  --> En-tête ignoré")
            continue
            
        parts = line.split('\t')
        if len(parts) >= 3:
            arc_name = parts[0].strip()
            section_id = parts[1].strip()  
            length_str = parts[2].strip()
            
            print(f"  --> Parsé: arc='{arc_name}', section='{section_id}', length='{length_str}'")
            
            if arc_name and '-' in arc_name:
                current_arc = f"{arc_name}"
                print(f"      --> Nouvel arc: {current_arc}")
            
            if section_id and length_str:
                print(f"      --> Ajout tronçon {section_id} à l'arc {current_arc}")

if __name__ == "__main__":
    debug_parse()