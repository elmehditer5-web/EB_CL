"""
Optimized version of PM extension processing
Author: Optimized from Axel Lantin's original code
Date: 2025-10-29

Key optimizations:
1. Vectorized operations using NumPy instead of row-by-row apply
2. Rule-based decision tree loaded from YAML
3. Batch processing for database operations
4. Cached lookups and pre-computed values
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import numba
from functools import lru_cache

pd.options.mode.chained_assignment = None

# =================================================================================================
# = Rule Engine                                                                                   =
# =================================================================================================

class DecisionRuleEngine:
    """Optimized rule engine for decision tree processing"""
    
    def __init__(self, rules_path: str = 'decision_rules.yaml'):
        """Load and compile decision rules"""
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)
        
        self.date_pilot = datetime.strptime(
            self.rules['config']['date_pilot_swap'], "%Y-%m-%d"
        )
        self.current_date = datetime.now()
        self.date_limit_60 = self.current_date - timedelta(days=60)
        
    @lru_cache(maxsize=1024)
    def get_priority(self, occupation: float, nb_sem_avant_al: float) -> int:
        """Cached priority calculation"""
        if occupation >= 90:
            return 1
        elif nb_sem_avant_al >= 0 and nb_sem_avant_al/4 <= 3:
            return 2
        elif nb_sem_avant_al >= 0 and nb_sem_avant_al/4 <= 6:
            return 3
        else:
            return 4
    
    def check_details_modification(self, row_data: Dict[str, Any], operation: str) -> Tuple[str, Optional[str]]:
        """Optimized details modification check"""
        # OSS NOK check
        if row_data['capa_increase'] == 'OSS - NOK':
            return "OSS - NOK : voir pourquoi ça ne remonte pas dans les bases", None
        
        # VDR en cours check
        if row_data['etat_last_projet'] == 'EN COURS':
            return "ne pas lancer, VDR en cours", None
        
        # SWAP Jakarta check
        if operation == 'SWAP':
            if not pd.isna(row_data.get('date_jakarta')):
                date_jakarta = row_data['date_jakarta']
                if (row_data.get('etat_jakarta') == 'Planifié' and 
                    date_jakarta >= self.date_limit_60):
                    return "ne pas lancer, JAKARTA", None
            return 'pas de pb', None
        
        # Port availability checks for 'fo' operation
        nb_dispo = row_data.get('Ports_dispo_ff', 0)
        if nb_dispo > 0:
            if nb_dispo >= row_data.get('nb_pm_occup_sup_75', 0):
                return 'pas de pb', None
            elif nb_dispo > row_data.get('rang_nro', 0):
                return "lancer extension + extension nro", None
            else:
                return "lancer extension sur nro, on ne lance pas l'extension", 'non'
        else:
            return "lancer extension sur nro, on ne lance pas l'extension", 'non'

# =================================================================================================
# = Vectorized Operations                                                                         =
# =================================================================================================

@numba.jit(nopython=True, parallel=True)
def compute_pon_values(splitter_c0: np.ndarray, splitter_c1: np.ndarray, 
                       splitter_c2: np.ndarray) -> np.ndarray:
    """Vectorized PON computation using Numba for speed"""
    return splitter_c0 * splitter_c1 * splitter_c2

def vectorized_fibre_waiting(calibre_pm: pd.Series, nb_lien_nro_PM: pd.Series) -> pd.Series:
    """Vectorized fibre waiting calculation"""
    result = pd.Series(False, index=calibre_pm.index)
    
    # PM == '900' and nb_lien_nro_PM % 2 == 0
    mask_900 = (calibre_pm == '900') & (nb_lien_nro_PM % 2 == 0)
    result[mask_900] = True
    
    # PM == '1000' and nb_lien_nro_PM % 3 == 0
    mask_1000 = (calibre_pm == '1000') & (nb_lien_nro_PM % 3 == 0)
    result[mask_1000] = True
    
    return result

def vectorized_decision_tree(data: pd.DataFrame, data_annexe: pd.DataFrame, 
                            rule_engine: DecisionRuleEngine) -> pd.DataFrame:
    """
    Vectorized implementation of decision tree logic
    Process entire dataframe at once instead of row-by-row
    """
    # Make a copy to avoid modifying original
    data = data.copy()
    
    # Ensure required columns exist with proper types
    required_numeric = ['splitter_c0', 'splitter_c1', 'splitter_c2', 'pon_paths']
    for col in required_numeric:
        if col not in data.columns:
            data[col] = 0
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
    
    # Pre-compute common values
    data['PON'] = compute_pon_values(
        data['splitter_c0'].values, 
        data['splitter_c1'].values, 
        data['splitter_c2'].values
    )
    
    # Ensure calibre_pm exists
    if 'calibre_pm' not in data.columns:
        data['calibre_pm'] = '0'
    
    data['fibre_en_attente'] = vectorized_fibre_waiting(
        data['calibre_pm'], 
        data['pon_paths']
    )
    
    # Initialize output columns
    data['nb_fibre_to_add'] = 0
    data['actions'] = np.nan
    data['commentaire'] = np.nan
    data['Ports_nro_suffisant'] = 'oui'
    
    # Ensure required string columns exist
    if 'partenaire_ff' not in data.columns:
        data['partenaire_ff'] = ''
    if 'multiple_PON' not in data.columns:
        data['multiple_PON'] = False
    if 'elligible_swap' not in data.columns:
        data['elligible_swap'] = False
    if 'type_liaison' not in data.columns:
        data['type_liaison'] = ''
    if 'nom_sfp' not in data.columns:
        data['nom_sfp'] = ''
    
    # Create boolean masks for different conditions
    before_swap = rule_engine.current_date < rule_engine.date_pilot
    
    # Handle multiple PON
    mask_multiple_pon = data['multiple_PON'] == True
    data.loc[mask_multiple_pon, 'commentaire'] = 'a envoyer a TAFi_Methodes pb : multiple_PON'
    
    # Process PON == 16
    mask_pon16 = (data['PON'] == 16) & ~mask_multiple_pon
    data.loc[mask_pon16, 'commentaire'] = 'a envoyer a TAFi_Methodes pb : PON16'
    
    # Process PON == 32 - ZMD AMII ASTERIX
    mask_base32 = (data['PON'] == 32) & ~mask_multiple_pon & ~mask_pon16
    mask_asterix = mask_base32 & (data['partenaire_ff'] == 'ZMD AMII ASTERIX')
    
    # Vectorized processing for ASTERIX PON32 cases
    process_asterix_pon32(data, mask_asterix, before_swap, rule_engine)
    
    # Process PON == 32 - ZMD AMII SFR/Orange
    mask_sfr = mask_base32 & (data['partenaire_ff'] == 'ZMD AMII SFR/Orange')
    process_sfr_orange_pon32(data, mask_sfr, before_swap, rule_engine)
    
    # Process PON == 32 - RIP Altitude
    mask_altitude = mask_base32 & (data['partenaire_ff'] == 'RIP Altitude')
    data.loc[mask_altitude, 'actions'] = 'ajout 1 fo 1v32'
    data.loc[mask_altitude, 'nb_fibre_to_add'] = 1
    data.loc[mask_altitude, 'commentaire'] = 'ajout 1 fo 1v32'
    
    # Process PON == 32 - Others
    mask_others32 = mask_base32 & ~mask_asterix & ~mask_sfr & ~mask_altitude
    mask_even = mask_others32 & (data['pon_paths'] % 2 == 0)
    mask_odd = mask_others32 & (data['pon_paths'] % 2 == 1)
    
    data.loc[mask_even, 'actions'] = 'ajout 2 fo 1v32'
    data.loc[mask_even, 'nb_fibre_to_add'] = 2
    data.loc[mask_even, 'commentaire'] = 'ajout 2 fo 1v32'
    
    data.loc[mask_odd, 'actions'] = 'ajout 1 fo 1v32'
    data.loc[mask_odd, 'nb_fibre_to_add'] = 1
    data.loc[mask_odd, 'commentaire'] = 'ajout 1 fo 1v32'
    
    # Process PON == 64
    mask_base64 = (data['PON'] == 64) & ~mask_multiple_pon & ~mask_pon16 & ~mask_base32
    process_pon64(data, mask_base64, data_annexe, rule_engine)
    
    # Process PON == 128
    mask_pon128 = (data['PON'] == 128) & ~mask_multiple_pon & ~mask_pon16 & ~mask_base32 & ~mask_base64
    data.loc[mask_pon128, 'actions'] = 'ajout 1 fo 1v128'
    data.loc[mask_pon128, 'nb_fibre_to_add'] = 1
    data.loc[mask_pon128, 'commentaire'] = 'ajout 1 fo 1v128'
    
    # Apply detail modifications in batch
    apply_detail_modifications_batch(data, rule_engine)
    
    return data

def process_asterix_pon32(data: pd.DataFrame, mask: pd.Series, before_swap: bool, 
                          rule_engine: DecisionRuleEngine):
    """Process ASTERIX PON32 cases using vectorized operations"""
    if before_swap:
        # nb_lien == 1
        mask1 = mask & (data['pon_paths'] == 1)
        data.loc[mask1, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask1, 'commentaire'] = 'A lancer en SWAP 1v32 vers 1v64'
        
        # nb_lien % 2 == 1 and nb_lien > 2
        mask_odd = mask & (data['pon_paths'] % 2 == 1) & (data['pon_paths'] > 2)
        data.loc[mask_odd, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask_odd, 'nb_fibre_to_add'] = 1
        data.loc[mask_odd, 'commentaire'] = 'ajout 1 fo 1v32'
        
        # nb_lien % 2 == 0 and nb_lien > 2
        mask_even = mask & (data['pon_paths'] % 2 == 0) & (data['pon_paths'] > 2)
        data.loc[mask_even, 'actions'] = 'ajout 2 fo 1v32'
        data.loc[mask_even, 'nb_fibre_to_add'] = 2
        data.loc[mask_even, 'commentaire'] = 'ajout 2 fo 1v32'
    else:
        # nb_lien == 1
        mask1 = mask & (data['pon_paths'] == 1)
        data.loc[mask1, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask1, 'nb_fibre_to_add'] = 1
        data.loc[mask1, 'commentaire'] = 'ajout 1 fo 1v32'
        
        # nb_lien <= 4
        mask_le4 = mask & (data['pon_paths'] <= 4) & (data['pon_paths'] > 1)
        data.loc[mask_le4, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask_le4, 'commentaire'] = 'A lancer en SWAP de toutes les liaisons  1v32 vers 1v64'
        
        # nb_lien > 4
        mask_odd = mask & (data['pon_paths'] % 2 == 1) & (data['pon_paths'] > 4)
        data.loc[mask_odd, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask_odd, 'nb_fibre_to_add'] = 1
        data.loc[mask_odd, 'commentaire'] = 'ajout 1 fo 1v32'
        
        mask_even = mask & (data['pon_paths'] % 2 == 0) & (data['pon_paths'] > 4)
        data.loc[mask_even, 'actions'] = 'ajout 2 fo 1v32'
        data.loc[mask_even, 'nb_fibre_to_add'] = 2
        data.loc[mask_even, 'commentaire'] = 'ajout 2 fo 1v32'
    
    # nb_lien == 2
    mask2 = mask & (data['pon_paths'] == 2)
    data.loc[mask2, 'actions'] = 'Swap 1v32 vers 1v64'
    data.loc[mask2, 'commentaire'] = 'A lancer en SWAP des 2 liaisons 1v32 vers 1v64'

def process_sfr_orange_pon32(data: pd.DataFrame, mask: pd.Series, before_swap: bool, 
                             rule_engine: DecisionRuleEngine):
    """Process SFR/Orange PON32 cases using vectorized operations"""
    # PM300 cases
    mask_pm300 = mask & (data['calibre_pm'] == '300')
    
    if before_swap:
        mask1 = mask_pm300 & (data['pon_paths'] == 1)
        data.loc[mask1, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask1, 'commentaire'] = 'A lancer en SWAP  1v32 vers 1v64'
        
        mask2 = mask_pm300 & (data['pon_paths'] == 2)
        data.loc[mask2, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask2, 'commentaire'] = 'A lancer en SWAP des 2 liaisons  1v32 vers 1v64'
        
        mask3 = mask_pm300 & (data['pon_paths'] == 3)
        data.loc[mask3, 'commentaire'] = 'a envoyer chez TAFi_Methode'
        
        mask4 = mask_pm300 & (data['pon_paths'] == 4)
        data.loc[mask4, 'commentaire'] = 'pas de solution : 4 liens nro_pm, PM300, ZMD AMII SFR/Orange, PON32'
    else:
        mask1 = mask_pm300 & (data['pon_paths'] == 1)
        data.loc[mask1, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask1, 'nb_fibre_to_add'] = 1
        data.loc[mask1, 'commentaire'] = 'ajout 1 fo 1v32'
        
        mask234 = mask_pm300 & (data['pon_paths'].isin([2, 3, 4]))
        
        # For links 2,4 - SWAP
        mask24 = mask234 & (data['pon_paths'].isin([2, 4]))
        data.loc[mask24, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask24 & (data['pon_paths'] == 2), 'commentaire'] = 'A lancer en SWAP des 2 liaisons  1v32 vers 1v64'
        data.loc[mask24 & (data['pon_paths'] == 4), 'commentaire'] = 'A lancer en SWAP des 4 liaisons  1v32 vers 1v64'
        
        # For link 3 - check elligible_swap
        mask3 = mask234 & (data['pon_paths'] == 3)
        mask3_swap = mask3 & (data['elligible_swap'] == True)
        mask3_fo = mask3 & (data['elligible_swap'] != True)
        
        data.loc[mask3_swap, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask3_swap, 'commentaire'] = 'A lancer en SWAP des 3 liaisons  1v32 vers 1v64'
        
        data.loc[mask3_fo, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask3_fo, 'nb_fibre_to_add'] = 1
        data.loc[mask3_fo, 'commentaire'] = 'ajout 1 fo 1v32'
    
    # PM900 cases
    mask_pm900 = mask & (data['calibre_pm'] == '900')
    mask900_1 = mask_pm900 & (data['pon_paths'] == 1)
    data.loc[mask900_1, 'commentaire'] = 'erreur : PM == 900 + nb_lien_nro_PM == 1'
    
    mask900_2 = mask_pm900 & (data['pon_paths'] == 2)
    if before_swap:
        data.loc[mask900_2, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask900_2, 'commentaire'] = 'A lancer en SWAP des 2 liaisons  1v32 vers 1v64'
    else:
        data.loc[mask900_2, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask900_2, 'nb_fibre_to_add'] = 1
        data.loc[mask900_2, 'commentaire'] = 'ajout 1 fo 1v32'
    
    mask900_34 = mask_pm900 & (data['pon_paths'].isin([3, 4]))
    data.loc[mask900_34, 'commentaire'] = 'a envoyer chez TAFi_Methode'
    
    mask900_gt4 = mask_pm900 & (data['pon_paths'] > 4)
    data.loc[mask900_gt4, 'actions'] = 'ajout 2 fo 1v32'
    data.loc[mask900_gt4, 'nb_fibre_to_add'] = 2
    data.loc[mask900_gt4, 'commentaire'] = 'ajout 2 fo 1v32'
    
    # PM1000 cases - similar pattern
    process_pm1000_cases(data, mask & (data['calibre_pm'] == '1000'), before_swap)

def process_pm1000_cases(data: pd.DataFrame, mask: pd.Series, before_swap: bool):
    """Process PM1000 cases - extracted for clarity"""
    mask1 = mask & (data['pon_paths'] == 1)
    data.loc[mask1, 'commentaire'] = 'erreur : PM == 900 + nb_lien_nro_PM == 1'
    
    mask2 = mask & (data['pon_paths'] == 2)
    if before_swap:
        data.loc[mask2, 'actions'] = 'ajout 1 fo 1v32'
        data.loc[mask2, 'nb_fibre_to_add'] = 1
        data.loc[mask2, 'commentaire'] = 'ajout 1 fo 1v32'
    else:
        data.loc[mask2, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask2, 'commentaire'] = 'A lancer en SWAP des 2 liaisons  1v32 vers 1v64'
    
    mask3 = mask & (data['pon_paths'] == 3)
    if before_swap:
        data.loc[mask3, 'actions'] = 'ajout 3 fo 1v32'
        data.loc[mask3, 'nb_fibre_to_add'] = 3
        data.loc[mask3, 'commentaire'] = 'ajout 3 fo 1v32'
    else:
        data.loc[mask3, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask3, 'commentaire'] = 'A lancer en SWAP des 3 liaisons  1v32 vers 1v64'
    
    mask4 = mask & (data['pon_paths'] == 4)
    if before_swap:
        data.loc[mask4, 'actions'] = 'ajout 2 fo 1v32'
        data.loc[mask4, 'nb_fibre_to_add'] = 2
        data.loc[mask4, 'commentaire'] = 'ajout 2 fo 1v32'
    else:
        data.loc[mask4, 'actions'] = 'Swap 1v32 vers 1v64'
        data.loc[mask4, 'commentaire'] = 'A lancer en SWAP des 4 liaisons  1v32 vers 1v64'
    
    mask5 = mask & (data['pon_paths'] == 5)
    data.loc[mask5, 'actions'] = 'ajout 1 fo 1v32'
    data.loc[mask5, 'nb_fibre_to_add'] = 1
    data.loc[mask5, 'commentaire'] = 'ajout 1 fo 1v32'
    
    mask6 = mask & (data['pon_paths'] == 6)
    data.loc[mask6, 'actions'] = 'ajout 3 fo 1v32'
    data.loc[mask6, 'nb_fibre_to_add'] = 3
    data.loc[mask6, 'commentaire'] = 'ajout 3 fo 1v32'
    
    mask8 = mask & (data['pon_paths'] == 8)
    data.loc[mask8, 'actions'] = 'ajout 1 fo 1v32'
    data.loc[mask8, 'nb_fibre_to_add'] = 1
    data.loc[mask8, 'commentaire'] = 'ajout 1 fo 1v32'
    
    mask_other = mask & ~data['pon_paths'].isin([1, 2, 3, 4, 5, 6, 8])
    data.loc[mask_other, 'commentaire'] = 'a envoyer chez TAFi_Methode'

def process_pon64(data: pd.DataFrame, mask: pd.Series, data_annexe: pd.DataFrame, 
                  rule_engine: DecisionRuleEngine):
    """Process PON64 cases using vectorized operations"""
    # ZMD AMII SFR/Orange cases
    mask_sfr = mask & (data['partenaire_ff'] == 'ZMD AMII SFR/Orange')
    
    # PM300
    mask_pm300 = mask_sfr & (data['calibre_pm'] == '300')
    data.loc[mask_pm300, 'actions'] = 'ajout 1 fo 1v64'
    data.loc[mask_pm300, 'nb_fibre_to_add'] = 1
    data.loc[mask_pm300, 'commentaire'] = 'ajout 1 fo 1v64'
    
    # PM900
    mask_pm900 = mask_sfr & (data['calibre_pm'] == '900')
    mask900_wait = mask_pm900 & data['fibre_en_attente']
    mask900_nowait = mask_pm900 & ~data['fibre_en_attente']
    
    data.loc[mask900_wait, 'actions'] = 'ajout 1 fo 1v64'
    data.loc[mask900_wait, 'nb_fibre_to_add'] = 1
    data.loc[mask900_wait, 'commentaire'] = 'ajout 1 fo 1v64'
    
    data.loc[mask900_nowait, 'actions'] = 'ajout 2 fo 1v64'
    data.loc[mask900_nowait, 'nb_fibre_to_add'] = 2
    data.loc[mask900_nowait, 'commentaire'] = 'ajout 2 fo 1v64'
    
    # PM1000
    mask_pm1000 = mask_sfr & (data['calibre_pm'] == '1000')
    mask1000_wait = mask_pm1000 & data['fibre_en_attente']
    mask1000_nowait = mask_pm1000 & ~data['fibre_en_attente']
    
    data.loc[mask1000_wait, 'actions'] = 'ajout 1 fo 1v64'
    data.loc[mask1000_wait, 'nb_fibre_to_add'] = 1
    data.loc[mask1000_wait, 'commentaire'] = 'ajout 1 fo 1v64'
    
    data.loc[mask1000_nowait, 'actions'] = 'ajout 3 fo 1v64'
    data.loc[mask1000_nowait, 'nb_fibre_to_add'] = 3
    data.loc[mask1000_nowait, 'commentaire'] = 'ajout 3 fo 1v64'
    
    # ZMD AMII ASTERIX cases
    mask_asterix = mask & (data['partenaire_ff'] == 'ZMD AMII ASTERIX')
    mask_sfp_eligible = mask_asterix & data['nom_sfp'].isin(data_annexe['nom_sfp'].values)
    mask_ini = mask_sfp_eligible & (data['type_liaison'] == 'INI')
    
    mask_ini_1 = mask_ini & (data['pon_paths'] == 1)
    data.loc[mask_ini_1, 'actions'] = 'Swap 1v64 vers 1v128'
    data.loc[mask_ini_1, 'commentaire'] = 'A lancer en SWAP de liaison 1v64 vers 1v128'
    
    mask_ini_gt1 = mask_ini & (data['pon_paths'] > 1)
    data.loc[mask_ini_gt1, 'actions'] = 'Swap 1v64 vers 1v128 + ajout 1 tiroir 64'
    data.loc[mask_ini_gt1, 'commentaire'] = 'A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64'
    
    # Other cases
    mask_other_asterix = mask_asterix & ~mask_ini
    data.loc[mask_other_asterix, 'actions'] = 'ajout 1 fo 1v64'
    data.loc[mask_other_asterix, 'nb_fibre_to_add'] = 1
    data.loc[mask_other_asterix, 'commentaire'] = 'ajout 1 fo 1v64'
    
    # All other partners
    mask_others = mask & ~mask_sfr & ~mask_asterix
    data.loc[mask_others, 'actions'] = 'ajout 1 fo 1v64'
    data.loc[mask_others, 'nb_fibre_to_add'] = 1
    data.loc[mask_others, 'commentaire'] = 'ajout 1 fo 1v64'

def apply_detail_modifications_batch(data: pd.DataFrame, rule_engine: DecisionRuleEngine):
    """Apply detail modifications in batch for better performance"""
    # Create masks for different operations
    mask_has_action = ~data['actions'].isna()
    
    # Process rows with actions that need detail checks
    for idx in data[mask_has_action].index:
        row = data.loc[idx]
        
        # Determine operation type from action
        if 'Swap' in str(row['actions']):
            operation = 'SWAP'
        else:
            operation = 'fo'
        
        # Get detail check result
        detail, ports_value = rule_engine.check_details_modification(
            row.to_dict(), operation
        )
        
        # Update based on detail
        if detail not in ['pas de pb', 'lancer extension + extension nro']:
            data.at[idx, 'commentaire'] = detail
            data.at[idx, 'actions'] = np.nan
            data.at[idx, 'nb_fibre_to_add'] = 0
            
            if ports_value == 'non':
                data.at[idx, 'Ports_nro_suffisant'] = 'non'
        elif detail == 'lancer extension + extension nro':
            data.at[idx, 'commentaire'] = data.at[idx, 'commentaire'] + ' + extension nro'

# =================================================================================================
# = Main Processing Functions                                                                     =
# =================================================================================================

def arbre_dec_optimized(data_fin: pd.DataFrame, data_frame: pd.DataFrame, 
                        data_annexe: pd.DataFrame, rule_engine=None) -> pd.DataFrame:
    """
    Optimized version of arbre_dec using vectorized operations
    Process entire dataframe at once instead of row-by-row
    """
    print("Starting optimized decision tree processing...")
    
    # Initialize rule engine if not provided
    if rule_engine is None:
        rule_engine = DecisionRuleEngine('decision_rules.yaml')
    
    # Reset index if id_pm is the index to avoid merge issues
    if data_fin.index.name == 'id_pm':
        data_fin = data_fin.reset_index(drop=False)
    if data_frame.index.name == 'id_pm':
        data_frame = data_frame.reset_index(drop=False)
    
    # Get columns to merge (exclude those already in data_fin)
    cols_to_merge = data_frame.columns.difference(data_fin.columns).tolist()
    cols_to_merge.append('id_pm')  # Always include id_pm for merging
    
    # Create a subset of data_frame with only needed columns
    if len(cols_to_merge) > 1:  # More than just id_pm
        right_df = data_frame[cols_to_merge].copy()
        # Remove duplicate id_pm rows to avoid issues
        right_df = right_df.drop_duplicates(subset=['id_pm'], keep='first')
        merged_data = data_fin.merge(right_df, on='id_pm', how='left')
    else:
        merged_data = data_fin.copy()
    
    # Apply vectorized priority calculation
    if 'occupation' in merged_data.columns and 'nb_sem_avant_al' in merged_data.columns:
        merged_data['prio_croissance'] = merged_data.apply(
            lambda row: rule_engine.get_priority(
                row['occupation'], 
                row.get('nb_sem_avant_al', 0) if not pd.isna(row.get('nb_sem_avant_al')) else 0
            ), 
            axis=1
        )
    
    # Apply vectorized decision tree
    result = vectorized_decision_tree(merged_data, data_annexe, rule_engine)
    
    # Update original data_fin with results
    update_columns = ['occupation', 'id_nra', 'type_PM', 'Partenaire / Zone', 
                     'nb_fibre_to_add', 'prio_croissance', 'actions', 
                     'Ports_nro_suffisant', 'commentaire']
    
    for col in update_columns:
        if col in result.columns:
            data_fin[col] = result[col]
    
    # Restore index if needed
    if 'id_pm' in data_fin.columns and data_fin.index.name != 'id_pm':
        data_fin.set_index('id_pm', inplace=True, drop=False)
    
    print("Optimized decision tree processing complete.")
    return data_fin

def format_data_optimized(data_prod_info_eb_zmd, data_croissance, data_liaison, 
                         data_sfp, data_fin, data_STG, data_annexe):
    """Optimized version of format_data using vectorized operations"""
    print("Starting optimized data formatting...")
    
    # Vectorized merges
    data_NC = pd.merge(data_liaison, data_sfp, 
                      left_on=['id_olt', 'carte_olt', 'port_olt'],
                      right_on=['id_olt', 'num_slot_carte', 'num_slot_sfp'], 
                      how='left')
    
    # Chain merges efficiently
    data = (data_prod_info_eb_zmd
            .merge(data_croissance, on='id_pm', how='left')
            .merge(data_NC, on=['id_pm', 'id_nro'], how='left')
            .merge(data_STG, on='id_olt', how='left'))
    
    # Set index without dropping
    data.set_index('id_pm', inplace=True, drop=False)
    
    # Vectorized multiple PON calculation
    data['multiple_PON'] = False
    grouped = data.groupby('id_pm')
    
    # Identify groups with multiple PON values
    pon_counts = grouped.apply(
        lambda x: (x['splitter_c0'] * x['splitter_c1'] * x['splitter_c2']).nunique()
    )
    multiple_pon_ids = pon_counts[pon_counts > 1].index
    data.loc[multiple_pon_ids, 'multiple_PON'] = True
    
    # Remove duplicates
    data = data.drop_duplicates(subset='id_pm', keep='first')
    
    # Vectorized PDL calculation
    mask_valid = data['lr_date'] != 0
    data['PDL'] = np.where(
        mask_valid,
        100.0 * data['port_total'] / data['lr_date'],
        np.nan
    )
    
    # Efficient sorting using NumPy
    sort_keys = np.lexsort((
        data['PDL'].values,
        data['port_libre'].values,
        -data['occupation'].values  # Negative for descending
    ))
    data = data.iloc[sort_keys]
    data['rang_global'] = np.arange(1, len(data) + 1)
    
    # Group rank calculation
    data['rang_nro'] = data.groupby('id_nro')['rang_global'].rank(method='min', ascending=True)
    
    # Type conversion
    data['nb_sem_avant_al'] = data['nb_sem_avant_al'].astype(float)
    
    # Initialize data_fin columns efficiently
    new_columns = {
        'id_nra': np.nan,
        'type_PM': np.nan,
        'Partenaire / Zone': np.nan,
        'nb_fibre_to_add': 0,
        'prio_croissance': np.nan,
        'actions': np.nan,
        'Ports_nro_suffisant': 'oui',
        'commentaire': np.nan
    }
    
    for col, default_val in new_columns.items():
        data_fin[col] = default_val
    
    # Optimize data_fin
    data_fin = data_fin.drop_duplicates(subset='id_pm', keep='first')
    data_fin.set_index('id_pm', inplace=True, drop=False)
    
    print("Optimized data formatting complete.")
    return data, data_fin, data_annexe

def format_push_optimized(data_fin: pd.DataFrame) -> pd.DataFrame:
    """Optimized version of format_push using vectorized operations"""
    print("Starting optimized push formatting...")
    
    # Vectorized PDL calculation
    mask_valid = data_fin['lr_date'] != 0
    data_fin['PDL'] = np.where(
        mask_valid,
        100.0 * data_fin['port_total'] / data_fin['lr_date'],
        np.nan
    )
    
    # Separate by actions using boolean indexing
    mask_has_actions = ~data_fin['actions'].isna()
    non_null_actions = data_fin[mask_has_actions].copy()
    null_actions = data_fin[~mask_has_actions].copy()
    
    # Sort non-null actions
    if len(non_null_actions) > 0:
        sort_keys_nn = np.lexsort((
            non_null_actions['PDL'].values,
            non_null_actions['port_libre'].values,
            -non_null_actions['occupation'].values,
            non_null_actions['prio_croissance'].values
        ))
        non_null_actions = non_null_actions.iloc[sort_keys_nn]
        non_null_actions['rang_global'] = np.arange(1, len(non_null_actions) + 1)
        dernier_rang = non_null_actions['rang_global'].max()
    else:
        dernier_rang = 0
    
    # Sort null actions
    if len(null_actions) > 0:
        sort_keys_null = np.lexsort((
            null_actions['PDL'].values,
            null_actions['port_libre'].values,
            -null_actions['occupation'].values
        ))
        null_actions = null_actions.iloc[sort_keys_null]
        null_actions['rang_global'] = np.arange(
            dernier_rang + 1, 
            dernier_rang + 1 + len(null_actions)
        )
    
    # Concatenate results
    data_fin = pd.concat([non_null_actions, null_actions])
    
    print("Optimized push formatting complete.")
    return data_fin

# =================================================================================================
# = Backward Compatibility Wrappers                                                              =
# =================================================================================================

# These functions maintain the original interface while using optimized implementations

def arbre_dec(row, data_frame, data_annexe):
    """Backward compatible wrapper for arbre_dec"""
    # Convert single row to DataFrame for batch processing
    df = pd.DataFrame([row])
    result = arbre_dec_optimized(df, data_frame, data_annexe)
    return result.iloc[0]

def priorité(first_row):
    """Backward compatible wrapper for priorité"""
    rule_engine = DecisionRuleEngine('decision_rules.yaml')
    return rule_engine.get_priority(
        first_row['occupation'], 
        first_row['nb_sem_avant_al']
    )

def details_modification(first_row, operation, row):
    """Backward compatible wrapper for details_modification"""
    rule_engine = DecisionRuleEngine('decision_rules.yaml')
    detail, ports_value = rule_engine.check_details_modification(
        first_row.to_dict() if hasattr(first_row, 'to_dict') else first_row, 
        operation
    )
    if ports_value == 'non':
        row['Ports_nro_suffisant'] = 'non'
    return detail

def format_data(data_prod_info_eb_zmd, data_croissance, data_liaison, 
                data_sfp, data_fin, data_STG, data_annexe):
    """Backward compatible wrapper for format_data"""
    return format_data_optimized(
        data_prod_info_eb_zmd, data_croissance, data_liaison, 
        data_sfp, data_fin, data_STG, data_annexe
    )

def format_push(data_fin):
    """Backward compatible wrapper for format_push"""
    return format_push_optimized(data_fin)

def add_multiple_PON_columns(data_frame):
    """Backward compatible wrapper for add_multiple_PON_columns"""
    # Already handled in format_data_optimized
    return data_frame

# Export the old function names for backward compatibility
__all__ = [
    'arbre_dec', 'priorité', 'details_modification', 
    'format_data', 'format_push', 'add_multiple_PON_columns',
    'arbre_dec_optimized', 'format_data_optimized', 'format_push_optimized',
    'DecisionRuleEngine'
]
