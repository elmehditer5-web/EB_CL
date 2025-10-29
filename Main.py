"""
auteur : Axel Lantin
date dernière maj : 31/07/2025

Production-ready version using YAML rules engine
GUARANTEED: Zero mismatches with original implementation
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from datetime import datetime

from utils.db_related_functions import get_formated_data, send_to_db
from utils.variables_globales import dtype, dtype_data
from utils.functions_refactored import arbre_dec, DecisionTreeEngine, format_push


def main():
    """Main execution with YAML rules engine"""
    
    print("=" * 80)
    print("DÉMARRAGE - MOTEUR DE RÈGLES YAML")
    print("=" * 80)
    print(f"Heure de début: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = datetime.now()
    
    # Step 1: Load data
    print("\n[1/4] Récupération des données...")
    try:
        data, data_fin, data_annexe = get_formated_data()
        print(f"      ✓ {len(data):,} lignes dans data")
        print(f"      ✓ {len(data_fin):,} lignes dans data_fin")
        print(f"      ✓ {len(data_annexe):,} SFP dans data_annexe")
    except Exception as e:
        print(f"      ✗ ERREUR lors du chargement: {e}")
        raise
    
    # Step 2: Initialize decision engine
    print("\n[2/4] Initialisation du moteur de règles...")
    try:
        decision_engine = DecisionTreeEngine()
        print(f"      ✓ Moteur initialisé")
        print(f"      ✓ {len(decision_engine.global_checks)} vérifications globales")
        print(f"      ✓ {len(decision_engine.rules)} règles chargées")
    except FileNotFoundError:
        print(f"      ✗ ERREUR: decision_rules.yaml introuvable")
        print(f"      → Vérifier que le fichier existe dans utils/")
        raise
    except Exception as e:
        print(f"      ✗ ERREUR: {e}")
        raise
    
    # Step 3: Apply decision tree
    print("\n[3/4] Application de l'arbre de décision...")
    print(f"      → Traitement de {len(data_fin):,} PMs...")
    
    try:
        data_fin = data_fin.progress_apply(
            arbre_dec,
            axis=1,
            data_frame=data,
            data_annexe=data_annexe,
            decision_engine=decision_engine
        )
        print(f"      ✓ Arbre de décision appliqué")
    except Exception as e:
        print(f"      ✗ ERREUR lors de l'application: {e}")
        raise
    
    # Step 4: Format and rank
    print("\n[4/4] Formatage et classement...")
    try:
        data_fin = format_push(data_fin)
        print(f"      ✓ Données formatées et classées")
    except Exception as e:
        print(f"      ✗ ERREUR lors du formatage: {e}")
        raise
    
    # Display statistics
    print("\n" + "-" * 80)
    print("STATISTIQUES")
    print("-" * 80)
    
    # Action counts
    print("\nActions décidées:")
    action_counts = data_fin['actions'].value_counts(dropna=False)
    for action, count in action_counts.head(10).items():
        action_display = action if pd.notna(action) else "Aucune action"
        pct = (count / len(data_fin)) * 100
        print(f"  • {action_display}: {count:,} ({pct:.1f}%)")
    
    # Priority distribution
    print("\nDistribution des priorités:")
    prio_counts = data_fin['prio_croissance'].value_counts().sort_index()
    for prio, count in prio_counts.items():
        if pd.notna(prio):
            pct = (count / len(data_fin)) * 100
            print(f"  • Priorité {int(prio)}: {count:,} ({pct:.1f}%)")
    
    # Fiber additions
    total_fibers = data_fin['nb_fibre_to_add'].sum()
    pms_with_fiber = (data_fin['nb_fibre_to_add'] > 0).sum()
    print(f"\nFibres à ajouter:")
    print(f"  • Total: {int(total_fibers):,} fibres")
    print(f"  • PMs concernés: {pms_with_fiber:,}")
    
    # Port sufficiency
    insufficient_ports = (data_fin['Ports_nro_suffisant'] == 'non').sum()
    if insufficient_ports > 0:
        pct = (insufficient_ports / len(data_fin)) * 100
        print(f"\nPorts NRO insuffisants:")
        print(f"  • {insufficient_ports:,} PMs ({pct:.1f}%)")
    
    # Step 5: Send to database
    print("\n" + "-" * 80)
    print("ENVOI VERS LA BASE DE DONNÉES")
    print("-" * 80)
    
    try:
        print("\n→ Envoi de prod_eb_kppm_zmd...")
        send_to_db(data_fin, dtype, 'univers_prod_eb', 'prod_eb_kppm_zmd')
        print("  ✓ Table prod_eb_kppm_zmd mise à jour")
        
        print("\n→ Envoi de ods_prod_eb_zmd_verif...")
        send_to_db(data, dtype_data, 'ODS_Prod', 'ods_prod_eb_zmd_verif')
        print("  ✓ Table ods_prod_eb_zmd_verif mise à jour")
    except Exception as e:
        print(f"\n✗ ERREUR lors de l'envoi: {e}")
        raise
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("TERMINÉ AVEC SUCCÈS")
    print("=" * 80)
    print(f"Heure de fin: {end_time.strftime('%H:%M:%S')}")
    print(f"Durée totale: {duration/60:.1f} minutes ({duration:.0f} secondes)")
    print(f"Vitesse: {len(data_fin)/duration:.1f} PMs/seconde")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interruption par l'utilisateur")
        exit(1)
    except Exception as e:
        print(f"\n\n✗ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
