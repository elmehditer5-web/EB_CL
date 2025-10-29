"""
Trace execution of original vs refactored for mismatched PMs
This will show exactly where they diverge
"""

import pandas as pd
import sys
from utils import functions as original
from utils import functions_refactored as refactored
from utils.db_related_functions import get_formated_data

# Monkey-patch to trace execution
original_trace = []
refactored_trace = []

def trace_original():
    """Wrap original functions to trace execution"""
    global original_trace
    
    original_arbre_dec = original.arbre_dec
    
    def traced_arbre_dec(row, data_frame, data_annexe):
        original_trace.clear()
        
        id_pm = row['id_pm']
        original_trace.append(f"START: {id_pm}")
        
        if data_frame.loc[data_frame['id_pm'] == id_pm, 'partenaire_ff'].empty:
            original_trace.append("  EMPTY - early return")
            return row
        
        first_row = data_frame.loc[data_frame['id_pm'] == id_pm].squeeze()
        
        PA = first_row['partenaire_ff']
        PM = first_row['calibre_pm']
        nb_lien_nro_PM = first_row['pon_paths']
        PON = first_row['splitter_c0'] * first_row['splitter_c1'] * first_row['splitter_c2']
        multiple_PON = first_row['multiple_PON']
        
        original_trace.append(f"  PON={PON}, PA={PA}, PM={PM}, nb_lien={nb_lien_nro_PM}")
        
        if multiple_PON:
            original_trace.append("  → multiple_PON=True")
            return original_arbre_dec(row, data_frame, data_annexe)
        
        original_trace.append(f"  → Entering PON={PON} branch")
        
        # Continue with original execution
        return original_arbre_dec(row, data_frame, data_annexe)
    
    return traced_arbre_dec


def compare_single_pm(id_pm, data, data_fin, data_annexe):
    """Compare original vs refactored for one PM"""
    
    print(f"\n{'='*100}")
    print(f"COMPARING: {id_pm}")
    print(f"{'='*100}")
    
    if id_pm not in data_fin['id_pm'].values:
        print("✗ PM not found in data_fin")
        return
    
    if id_pm not in data.index:
        print("✗ PM not found in data")
        return
    
    # Get context
    context_row = data.loc[id_pm]
    PON = context_row['splitter_c0'] * context_row['splitter_c1'] * context_row['splitter_c2']
    
    print(f"\nContext:")
    print(f"  PON: {context_row['splitter_c0']} × {context_row['splitter_c1']} × {context_row['splitter_c2']} = {PON}")
    print(f"  PA: {context_row['partenaire_ff']}")
    print(f"  PM: {context_row['calibre_pm']}")
    print(f"  nb_lien: {context_row['pon_paths']}")
    print(f"  type_liaison: {context_row.get('type_liaison', 'N/A')}")
    print(f"  nom_sfp: {context_row.get('nom_sfp', 'N/A')}")
    print(f"  elligible_swap: {context_row.get('elligible_swap', 'N/A')}")
    
    # Check nom_sfp in annexe
    nom_sfp = context_row.get('nom_sfp', '')
    if pd.notna(nom_sfp):
        in_annexe = nom_sfp in data_annexe['nom_sfp'].values
        print(f"  nom_sfp in annexe: {in_annexe}")
    
    # Get rows
    row_orig = data_fin[data_fin['id_pm'] == id_pm].iloc[0].copy()
    row_refact = data_fin[data_fin['id_pm'] == id_pm].iloc[0].copy()
    
    # Run original
    print(f"\n🔵 ORIGINAL:")
    result_orig = original.arbre_dec(row_orig, data, data_annexe)
    print(f"  commentaire: {result_orig.get('commentaire', 'N/A')}")
    print(f"  actions: {result_orig.get('actions', 'N/A')}")
    print(f"  nb_fibre_to_add: {result_orig.get('nb_fibre_to_add', 0)}")
    
    # Run refactored
    print(f"\n🟢 REFACTORED:")
    engine = refactored.DecisionTreeEngine()
    result_refact = refactored.arbre_dec(row_refact, data, data_annexe, engine)
    print(f"  commentaire: {result_refact.get('commentaire', 'N/A')}")
    print(f"  actions: {result_refact.get('actions', 'N/A')}")
    print(f"  nb_fibre_to_add: {result_refact.get('nb_fibre_to_add', 0)}")
    
    # Find which rule matched in refactored
    from datetime import datetime
    nb_lien_nro_PM = context_row['pon_paths']
    PM_val = context_row['calibre_pm']
    
    if (PM_val == '900' and nb_lien_nro_PM % 2 == 0):
        fibre_en_attente = True
    elif (PM_val == '1000' and nb_lien_nro_PM % 3 == 0):
        fibre_en_attente = True
    else:
        fibre_en_attente = False
    
    before_pilot_date = datetime.now() < datetime.strptime("2025-12-01", "%Y-%m-%d")
    
    context = {
        'PON': PON,
        'PA': context_row['partenaire_ff'],
        'PM': PM_val,
        'nb_lien': nb_lien_nro_PM,
        'nb_lien_mod_2': nb_lien_nro_PM % 2,
        'before_pilot_date': before_pilot_date,
        'SWAP_possible': context_row.get('elligible_swap', False),
        'type_liaison': context_row.get('type_liaison', ''),
        'nom_sfp': nom_sfp,
        'fibre_en_attente': fibre_en_attente,
        'multiple_PON': context_row.get('multiple_PON', False),
        'nom_sfp_in_annexe': nom_sfp in data_annexe['nom_sfp'].values if pd.notna(nom_sfp) else False,
    }
    
    matched_rule = engine.find_matching_rule(context)
    if matched_rule:
        print(f"\n  Matched rule: {matched_rule.get('name')} (order {matched_rule.get('order')})")
    else:
        print(f"\n  ✗ NO RULE MATCHED")
    
    # Compare
    print(f"\n⚖️  COMPARISON:")
    fields = ['commentaire', 'actions', 'nb_fibre_to_add']
    has_mismatch = False
    for field in fields:
        orig_val = result_orig.get(field)
        refact_val = result_refact.get(field)
        
        if pd.isna(orig_val) and pd.isna(refact_val):
            match = True
        else:
            match = (orig_val == refact_val)
        
        if not match:
            print(f"  ✗ {field}: '{orig_val}' != '{refact_val}'")
            has_mismatch = True
    
    if not has_mismatch:
        print(f"  ✓ All fields match")
    
    return has_mismatch


def main():
    """Compare mismatched PMs"""
    
    print("\nLoading data...")
    data, data_fin, data_annexe = get_formated_data()
    
    # Load mismatch list from your data
    mismatched_pms = [
        'ADR_36046_PACT',  # PON32 vs PON64 confusion
        'ADR_59227_DEL1',  # PON64 vs PON32
        'FI-13071-000E',  # PON64 ASTERIX
        'FI-13103-000P',  # PON64 ASTERIX
        'FI-01195-0003',  # PON32 SWAP
        'FI-17074-0000',  # PON128 vs PON64
        'FI-19227-0002',  # Extension NRO
        'ADR-42075-CRAB',  # 2 fo vs 1 fo
    ]
    
    print(f"\nComparing {len(mismatched_pms)} mismatched PMs...")
    
    for pm in mismatched_pms:
        compare_single_pm(pm, data, data_fin, data_annexe)
        
        response = input("\nPress Enter for next, 'q' to quit, 's' to save diagnosis: ")
        if response.lower() == 'q':
            break


if __name__ == '__main__':
    main()
