"""
Diagnose specific mismatches by running both versions on problem PMs
"""

import pandas as pd
from utils import functions as original
from utils import functions_refactored as refactored
from utils.db_related_functions import get_formated_data

def diagnose_single_pm(id_pm, data, data_fin, data_annexe):
    """Diagnose a single PM"""
    
    print(f"\n{'='*80}")
    print(f"Diagnosing: {id_pm}")
    print(f"{'='*80}")
    
    # Get the row
    if id_pm not in data_fin['id_pm'].values:
        print(f"✗ PM not found: {id_pm}")
        return
    
    row_orig = data_fin[data_fin['id_pm'] == id_pm].iloc[0].copy()
    row_refact = data_fin[data_fin['id_pm'] == id_pm].iloc[0].copy()
    
    # Get context data
    if id_pm in data.index:
        context_row = data.loc[id_pm]
        print(f"\nContext:")
        print(f"  PON: {context_row['splitter_c0']} * {context_row['splitter_c1']} * {context_row['splitter_c2']} = {context_row['splitter_c0'] * context_row['splitter_c1'] * context_row['splitter_c2']}")
        print(f"  PA: {context_row['partenaire_ff']}")
        print(f"  PM: {context_row['calibre_pm']}")
        print(f"  nb_lien: {context_row['pon_paths']}")
        print(f"  type_liaison: {context_row.get('type_liaison', 'N/A')}")
        print(f"  nom_sfp: {context_row.get('nom_sfp', 'N/A')}")
        print(f"  elligible_swap: {context_row.get('elligible_swap', 'N/A')}")
        
        # Check if nom_sfp in annexe
        if 'nom_sfp' in context_row and not pd.isna(context_row['nom_sfp']):
            in_annexe = context_row['nom_sfp'] in data_annexe['nom_sfp'].values
            print(f"  nom_sfp in annexe: {in_annexe}")
    
    # Run original
    print(f"\n Running ORIGINAL...")
    try:
        result_orig = original.arbre_dec(row_orig, data, data_annexe)
        print(f"  commentaire: {result_orig.get('commentaire', 'N/A')}")
        print(f"  actions: {result_orig.get('actions', 'N/A')}")
        print(f"  nb_fibre_to_add: {result_orig.get('nb_fibre_to_add', 0)}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_orig = row_orig
    
    # Run refactored
    print(f"\n Running REFACTORED...")
    try:
        engine = refactored.DecisionTreeEngine()
        result_refact = refactored.arbre_dec(row_refact, data, data_annexe, engine)
        print(f"  commentaire: {result_refact.get('commentaire', 'N/A')}")
        print(f"  actions: {result_refact.get('actions', 'N/A')}")
        print(f"  nb_fibre_to_add: {result_refact.get('nb_fibre_to_add', 0)}")
        
        # Show which rule matched
        if id_pm in data.index:
            context_row = data.loc[id_pm]
            PON = context_row['splitter_c0'] * context_row['splitter_c1'] * context_row['splitter_c2']
            PA = context_row['partenaire_ff']
            PM = context_row['calibre_pm']
            nb_lien = context_row['pon_paths']
            
            context = {
                'PON': PON,
                'PA': PA,
                'PM': PM,
                'nb_lien': nb_lien,
                'nb_lien_mod_2': nb_lien % 2,
                'before_pilot_date': True,  # Approximate
                'SWAP_possible': context_row.get('elligible_swap', False),
                'type_liaison': context_row.get('type_liaison', ''),
                'nom_sfp': context_row.get('nom_sfp', ''),
                'fibre_en_attente': False,  # Approximate
                'multiple_PON': context_row.get('multiple_PON', False),
                'nom_sfp_in_annexe': context_row.get('nom_sfp', '') in data_annexe['nom_sfp'].values if not pd.isna(context_row.get('nom_sfp', '')) else False,
            }
            
            matched = engine.find_matching_rule(context)
            if matched:
                print(f"  Matched rule: {matched.get('name', 'Unknown')}")
                print(f"  Order: {matched.get('order', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        result_refact = row_refact
    
    # Compare
    print(f"\n COMPARISON:")
    fields = ['commentaire', 'actions', 'nb_fibre_to_add']
    mismatch = False
    for field in fields:
        orig_val = result_orig.get(field, 'N/A')
        refact_val = result_refact.get(field, 'N/A')
        match = "✓" if orig_val == refact_val else "✗"
        if orig_val != refact_val:
            mismatch = True
        print(f"  {match} {field}:")
        print(f"      Original:    {orig_val}")
        print(f"      Refactored:  {refact_val}")
    
    if not mismatch:
        print(f"\n✓ No mismatches for this PM")
    else:
        print(f"\n✗ MISMATCHES FOUND")


def main():
    """Diagnose known problematic PMs"""
    
    print("Loading data...")
    data, data_fin, data_annexe = get_formated_data()
    print(f"Loaded {len(data)} context rows, {len(data_fin)} PMs")
    
    # Test cases from the mismatch report
    problem_pms = [
        'FI-02722-0003',  # ajout 1 fo 1v64 -> Swap 1v64 vers 1v128 + ajout 1 tiroir 64
        'FI-17074-0000',  # ajout 1 fo 1v128 -> ajout 1 fo 1v64
        'FI-31417-0001',  # ajout 1 fo 1v64 -> Swap 1v32 vers 1v64
        'ADR_36046_PACT', # ajout 1 fo 1v32 -> ajout 1 fo 1v64
    ]
    
    for pm in problem_pms:
        diagnose_single_pm(pm, data, data_fin, data_annexe)
        input("\nPress Enter to continue to next PM...")


if __name__ == '__main__':
    main()
