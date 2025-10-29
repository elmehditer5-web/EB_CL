"""
auteur : Axel Lantin
date dernière maj : 31/07/2025

ce code utilise la vue v_prod_info_eb_zmd_axel pour mettre à jour la table prod_pm_info_zmd
il est basé sur l'arbre de décision : ZMD - Arbre de décision ZMD (\\bt0d0000\PARTAGES\1301-1350\M01340\0.-Documentation\02- KP_FTTH\02- KP PM)

UPDATED: Refactored to use rule-based decision engine
"""

# =================================================================================================
# = Les imports                                                                                   =
# =================================================================================================

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm
tqdm.pandas()


from utils.db_related_functions import get_formated_data
from utils.db_related_functions import send_to_db
from utils.variables_globales import dtype, dtype_data, server, user, password

# Refactored decision engine
from utils.functions_refactored import arbre_dec, DecisionTreeEngine

# Use original format_push (unchanged)
from utils.functions import format_push


# =================================================================================================
# = le code                                                                                       =
# =================================================================================================


def main():
    """
    Main execution function using refactored decision tree engine.
    """
    
    print("=" * 80)
    print("DÉMARRAGE DU PROGRAMME")
    print("=" * 80)
    
    # Step 1: Retrieve and format data
    print("\n[1/4] Récupération des données...")
    data, data_fin, data_annexe = get_formated_data()
    print(f"      ✓ {len(data)} lignes dans data")
    print(f"      ✓ {len(data_fin)} lignes dans data_fin")
    print(f"      ✓ {len(data_annexe)} SFP éligibles dans data_annexe")
    
    # Step 2: Apply decision tree with refactored engine
    print("\n[2/4] Application de l'arbre de décision...")
    print("      → Utilisation du moteur de règles")
    
    # Create decision engine once (reused for all rows)
    decision_engine = DecisionTreeEngine()
    print(f"      ✓ Moteur de règles initialisé")
    print(f"      ✓ {len(decision_engine.rules)} règles chargées")
    
    # Apply decision tree to each row
    data_fin = data_fin.progress_apply(
        arbre_dec, 
        axis=1, 
        data_frame=data, 
        data_annexe=data_annexe,
        decision_engine=decision_engine
    )
    
    print("      ✓ Arbre de décision appliqué")
    
    # Step 3: Format results
    print("\n[3/4] Formatage des résultats...")
    data_fin = format_push(data_fin)
    print("      ✓ Données formatées et classées")
    
    # Display summary statistics
    print("\n      Résumé des actions:")
    action_counts = data_fin['actions'].value_counts(dropna=False)
    for action, count in action_counts.head(10).items():
        action_display = action if pd.notna(action) else "Aucune action"
        print(f"        • {action_display}: {count}")
    
    # Step 4: Send to database
    print("\n[4/4] Envoi vers la base de données...")
    
    # Send final results
    send_to_db(data_fin, dtype, 'univers_prod_eb', 'prod_eb_kppm_zmd')
    print("      ✓ Table prod_eb_kppm_zmd mise à jour")
    
    # Send verification data
    send_to_db(data, dtype_data, 'ODS_Prod', 'ods_prod_eb_zmd_verif')
    print("      ✓ Table ods_prod_eb_zmd_verif mise à jour")
    
    print("\n" + "=" * 80)
    print("FIN DES OPÉRATIONS - SUCCÈS")
    print("=" * 80)


if __name__ == '__main__':
    main()
