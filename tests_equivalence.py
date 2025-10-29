"""
Equivalence test harness - Verifies zero mismatches
Run with: python tests_equivalence.py [--sample N | --full]
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Import both versions
from utils import functions as original
from utils import functions_refactored as refactored
from utils.db_related_functions import get_formated_data


class EquivalenceTestHarness:
    """Test exact behavioral equivalence"""
    
    def __init__(self, sample_size=None):
        self.sample_size = sample_size
        self.mismatches = []
        self.fields_to_compare = [
            'rang_global',
            'commentaire',
            'actions',
            'nb_fibre_to_add',
            'Ports_nro_suffisant'
        ]
    
    def load_data(self):
        """Load test data"""
        print("\n" + "=" * 80)
        print("CHARGEMENT DES DONNÉES")
        print("=" * 80)
        
        data, data_fin, data_annexe = get_formated_data()
        
        if self.sample_size:
            # Stratified sample by PA and PON
            if 'partenaire_ff' in data_fin.columns:
                data_fin = data_fin.groupby('partenaire_ff', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), max(1, self.sample_size // 10)), random_state=42)
                )
            else:
                data_fin = data_fin.sample(n=min(self.sample_size, len(data_fin)), random_state=42)
            print(f"\n✓ Échantillon stratifié: {len(data_fin):,} lignes")
        else:
            print(f"\n✓ Jeu de données complet: {len(data_fin):,} lignes")
        
        return data, data_fin, data_annexe
    
    def run_original(self, data, data_fin, data_annexe):
        """Run original implementation"""
        print("\n" + "=" * 80)
        print("EXÉCUTION VERSION ORIGINALE")
        print("=" * 80)
        
        start = datetime.now()
        data_fin_orig = data_fin.copy()
        
        print("\n→ Application de arbre_dec original...")
        for idx, row in data_fin_orig.iterrows():
            data_fin_orig.loc[idx] = original.arbre_dec(row, data, data_annexe)
        
        print("→ Application de format_push original...")
        data_fin_orig = original.format_push(data_fin_orig)
        
        duration = (datetime.now() - start).total_seconds()
        print(f"\n✓ Version originale terminée en {duration:.1f}s")
        print(f"  Vitesse: {len(data_fin_orig)/duration:.1f} lignes/sec")
        
        return data_fin_orig, duration
    
    def run_refactored(self, data, data_fin, data_annexe):
        """Run refactored implementation"""
        print("\n" + "=" * 80)
        print("EXÉCUTION VERSION REFACTORISÉE")
        print("=" * 80)
        
        start = datetime.now()
        data_fin_refact = data_fin.copy()
        
        print("\n→ Initialisation du moteur de règles...")
        decision_engine = refactored.DecisionTreeEngine()
        print(f"  ✓ {len(decision_engine.rules)} règles chargées")
        
        print("\n→ Application de arbre_dec refactorisé...")
        for idx, row in data_fin_refact.iterrows():
            data_fin_refact.loc[idx] = refactored.arbre_dec(
                row, data, data_annexe, decision_engine
            )
        
        print("→ Application de format_push...")
        data_fin_refact = refactored.format_push(data_fin_refact)
        
        duration = (datetime.now() - start).total_seconds()
        print(f"\n✓ Version refactorisée terminée en {duration:.1f}s")
        print(f"  Vitesse: {len(data_fin_refact)/duration:.1f} lignes/sec")
        
        return data_fin_refact, duration
    
    def compare_results(self, orig, refact):
        """Compare and record mismatches"""
        print("\n" + "=" * 80)
        print("COMPARAISON DES RÉSULTATS")
        print("=" * 80)
        
        mismatch_counts = {}
        mismatch_details = []
        
        for field in self.fields_to_compare:
            # Handle NaN comparison
            mask_equal = (orig[field] == refact[field]) | (
                orig[field].isna() & refact[field].isna()
            )
            
            mismatches = ~mask_equal
            mismatch_count = mismatches.sum()
            mismatch_counts[field] = mismatch_count
            
            if mismatch_count > 0:
                print(f"\n✗ {field}: {mismatch_count:,} différences")
                
                # Record first 50 per field
                for idx in orig[mismatches].index[:50]:
                    mismatch_details.append({
                        'input_id': orig.loc[idx, 'id_pm'],
                        'column_name': field,
                        'original_value': str(orig.loc[idx, field])[:100],
                        'refactored_value': str(refact.loc[idx, field])[:100],
                        'PA': orig.loc[idx, 'partenaire_ff'] if 'partenaire_ff' in orig.columns else 'N/A',
                        'PM': str(orig.loc[idx, 'type_PM']) if 'type_PM' in orig.columns else 'N/A',
                        'occupation': orig.loc[idx, 'occupation'] if 'occupation' in orig.columns else 'N/A',
                    })
            else:
                print(f"\n✓ {field}: MATCH PARFAIT")
        
        self.mismatches = mismatch_details
        return mismatch_counts
    
    def save_reports(self, mismatch_counts):
        """Save detailed reports"""
        print("\n" + "=" * 80)
        print("SAUVEGARDE DES RAPPORTS")
        print("=" * 80)
        
        # Save CSV
        if self.mismatches:
            df_mism = pd.DataFrame(self.mismatches)
            df_mism.to_csv('mismatch_report.csv', index=False)
            print(f"\n✓ mismatch_report.csv ({len(self.mismatches)} lignes)")
        
        # Save markdown summary
        with open('mismatch_summary.md', 'w', encoding='utf-8') as f:
            f.write("# Rapport de Vérification d'Équivalence\n\n")
            f.write(f"Généré: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Résumé des Différences\n\n")
            f.write("| Colonne | Différences | Statut |\n")
            f.write("|---------|-------------|--------|\n")
            
            total = 0
            for field, count in mismatch_counts.items():
                status = "✓ MATCH" if count == 0 else "✗ DIFF"
                f.write(f"| {field} | {count:,} | {status} |\n")
                total += count
            
            f.write(f"\n**Total:** {total:,} différences\n\n")
            
            if total == 0:
                f.write("## ✓✓✓ ÉQUIVALENCE PARFAITE ✓✓✓\n\n")
                f.write("Les deux implémentations produisent des résultats identiques.\n")
            else:
                f.write("## Exemples de Différences (50 premiers)\n\n")
                for i, m in enumerate(self.mismatches[:50], 1):
                    f.write(f"### {i}. {m['input_id']} - {m['column_name']}\n\n")
                    f.write(f"- **Original:** `{m['original_value']}`\n")
                    f.write(f"- **Refactorisé:** `{m['refactored_value']}`\n")
                    f.write(f"- **PA:** {m['PA']}\n")
                    f.write(f"- **PM:** {m['PM']}\n")
                    f.write(f"- **Occupation:** {m['occupation']}\n\n")
        
        print(f"✓ mismatch_summary.md")
    
    def run(self):
        """Execute full test"""
        print("\n" + "=" * 80)
        print("TEST D'ÉQUIVALENCE COMPORTEMENTALE")
        print("=" * 80)
        
        # Load
        data, data_fin, data_annexe = self.load_data()
        
        # Run both
        orig, time_orig = self.run_original(data, data_fin, data_annexe)
        refact, time_refact = self.run_refactored(data, data_fin, data_annexe)
        
        # Compare
        mismatch_counts = self.compare_results(orig, refact)
        
        # Save
        self.save_reports(mismatch_counts)
        
        # Final summary
        print("\n" + "=" * 80)
        print("RÉSULTAT FINAL")
        print("=" * 80)
        
        total_mismatches = sum(mismatch_counts.values())
        
        print(f"\nLignes testées: {len(orig):,}")
        print(f"Colonnes comparées: {len(self.fields_to_compare)}")
        print(f"\nDifférences par colonne:")
        for field, count in mismatch_counts.items():
            status = "✓" if count == 0 else "✗"
            print(f"  {status} {field}: {count:,}")
        
        print(f"\n{'='*80}")
        print(f"TOTAL DES DIFFÉRENCES: {total_mismatches:,}")
        print(f"{'='*80}")
        
        print(f"\nPerformance:")
        print(f"  Original: {time_orig:.1f}s ({len(orig)/time_orig:.1f} lignes/sec)")
        print(f"  Refactorisé: {time_refact:.1f}s ({len(refact)/time_refact:.1f} lignes/sec)")
        speedup = time_orig / time_refact if time_refact > 0 else 1.0
        print(f"  Facteur: {speedup:.2f}x")
        
        if total_mismatches == 0:
            print("\n" + "=" * 80)
            print("✓✓✓ SUCCÈS - ÉQUIVALENCE PARFAITE ✓✓✓")
            print("=" * 80)
            return True
        else:
            print("\n" + "=" * 80)
            print(f"✗✗✗ ÉCHEC - {total_mismatches:,} DIFFÉRENCES ✗✗✗")
            print("=" * 80)
            print("\nConsulter:")
            print("  • mismatch_report.csv (détails)")
            print("  • mismatch_summary.md (résumé)")
            return False


def main():
    # Create a new parser that won't conflict with other argparse usage
    parser = argparse.ArgumentParser(
        description='Test d\'équivalence entre version originale et refactorisée',
        conflict_handler='resolve'  # This resolves conflicts with other argparse
    )
    parser.add_argument(
        '--sample', 
        type=int, 
        default=None,
        help='Taille d\'échantillon (défaut: 1000 pour test rapide)'
    )
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Tester sur le jeu de données complet'
    )
    
    # Parse only known args to avoid conflicts
    args, unknown = parser.parse_known_args()
    
    # Determine sample size
    if args.full:
        sample_size = None
        print("\n→ Mode: JEU COMPLET")
    elif args.sample:
        sample_size = args.sample
        print(f"\n→ Mode: ÉCHANTILLON ({sample_size:,} lignes)")
    else:
        sample_size = 1000
        print(f"\n→ Mode: ÉCHANTILLON PAR DÉFAUT ({sample_size:,} lignes)")
        print("   (Utiliser --full pour tester tout le jeu)")
    
    # Run test
    harness = EquivalenceTestHarness(sample_size=sample_size)
    success = harness.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
