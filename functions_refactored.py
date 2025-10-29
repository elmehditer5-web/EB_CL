"""
auteur : Axel Lantin (refactored)
date dernière maj : 31/07/2025

Refactored version with extracted decision tree
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import yaml
from pathlib import Path

# Global pilot date - kept as module-level for backward compatibility
date_str = "2025-12-01"


class DecisionTreeEngine:
    """Loads and executes decision rules from YAML"""
    
    def __init__(self, rules_path=None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "decision_rules.yaml"
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules_config = yaml.safe_load(f)
        
        self.global_checks = self.rules_config.get('global_checks', [])
        self.rules = self.rules_config.get('rules', [])
    
    def evaluate_condition(self, condition_str, context):
        """
        Safely evaluate a condition string with given context.
        Preserves exact behavior including operator precedence.
        """
        # Build safe evaluation namespace
        safe_dict = {
            'PON': context['PON'],
            'PA': context['PA'],
            'PM': context['PM'],
            'nb_lien': context['nb_lien'],
            'before_pilot_date': context['before_pilot_date'],
            'SWAP_possible': context['SWAP_possible'],
            'type_liaison': context['type_liaison'],
            'nom_sfp': context['nom_sfp'],
            'fibre_en_attente': context['fibre_en_attente'],
            'multiple_PON': context['multiple_PON'],
            'annexe': context['annexe'],
            'True': True,
            'False': False,
        }
        
        # Handle special 'in annexe' check
        if 'in annexe' in condition_str:
            return context['nom_sfp'] in context['annexe']
        if 'not in annexe' in condition_str:
            return context['nom_sfp'] not in context['annexe']
        
        try:
            return eval(condition_str, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{condition_str}': {e}")
    
    def match_nb_lien_range(self, rule_nb_lien, actual_nb_lien):
        """Check if nb_lien matches the rule specification"""
        if isinstance(rule_nb_lien, int):
            return actual_nb_lien == rule_nb_lien
        
        if isinstance(rule_nb_lien, str):
            if rule_nb_lien == "other":
                return True
            if rule_nb_lien == ">=4":
                return actual_nb_lien >= 4
            if rule_nb_lien == ">4":
                return actual_nb_lien > 4
            if rule_nb_lien == "3-4":
                return 3 <= actual_nb_lien <= 4
            if '-' in rule_nb_lien:
                start, end = map(int, rule_nb_lien.split('-'))
                return start <= actual_nb_lien <= end
        
        return False
    
    def apply_rules(self, context):
        """
        Apply decision rules to context.
        Returns: (action, operation, comment, nb_fibre) or None if no match
        """
        # Check global conditions first
        for check in self.global_checks:
            if self.evaluate_condition(check['condition'], context):
                return (
                    check['action'],
                    None,
                    check['comment'],
                    0,
                    check.get('stop', False)
                )
        
        # Main rule traversal
        for rule in self.rules:
            if not self._match_rule(rule, context):
                continue
            
            # Rule matched, extract action
            result = self._extract_action(rule, context)
            if result:
                return result
        
        return None
    
    def _match_rule(self, rule, context):
        """Check if a rule matches the context"""
        if 'pon' in rule and rule['pon'] != context['PON']:
            return False
        
        if 'pa' in rule:
            if rule['pa'] == 'other':
                # Match if not any of the specific PAs
                specific_pas = ['ZMD AMII ASTERIX', 'ZMD AMII SFR/Orange', 'RIP Altitude']
                if context['PA'] in specific_pas:
                    return False
            elif rule['pa'] != context['PA']:
                return False
        
        if 'pm' in rule and rule['pm'] != context['PM']:
            return False
        
        return True
    
    def _extract_action(self, rule, context):
        """Extract action from matched rule"""
        # Simple action
        if 'action' in rule and 'subrules' not in rule and 'check_parity' not in rule:
            return (
                rule['action'],
                rule.get('operation'),
                rule.get('comment'),
                rule.get('nb_fibre', 0),
                False
            )
        
        # Has subrules based on nb_lien
        if 'subrules' in rule:
            for subrule in rule['subrules']:
                if 'nb_lien' in subrule:
                    if not self.match_nb_lien_range(subrule['nb_lien'], context['nb_lien']):
                        continue
                
                # Check conditions within subrule
                if 'conditions' in subrule:
                    for cond in subrule['conditions']:
                        if self.evaluate_condition(cond['condition'], context):
                            return (
                                cond['action'],
                                cond.get('operation'),
                                cond.get('comment'),
                                cond.get('nb_fibre', 0),
                                False
                            )
                elif 'action' in subrule:
                    return (
                        subrule['action'],
                        subrule.get('operation'),
                        subrule.get('comment'),
                        subrule.get('nb_fibre', 0),
                        False
                    )
                
                # Check for fallthrough
                if 'fallthrough' in subrule and subrule['fallthrough'] == 'check_parity':
                    if 'check_parity' in rule:
                        return self._check_parity(rule['check_parity'], context)
        
        # Parity check
        if 'check_parity' in rule:
            return self._check_parity(rule['check_parity'], context)
        
        return None
    
    def _check_parity(self, parity_rules, context):
        """Evaluate parity-based rules"""
        for parity_rule in parity_rules:
            if self.evaluate_condition(parity_rule['condition'], context):
                return (
                    parity_rule['action'],
                    parity_rule.get('operation'),
                    parity_rule.get('comment'),
                    parity_rule.get('nb_fibre', 0),
                    False
                )
        return None


def priorité(first_row):
    """
    Calculate priority based on occupation and nb_sem_avant_al.
    Exact replica of original function.
    """
    nb_sem_av_al = first_row['nb_sem_avant_al']
    if first_row['occupation'] >= 90:
        return 1
    elif nb_sem_av_al / 4 <= 3 and nb_sem_av_al >= 0:
        return 2
    elif nb_sem_av_al / 4 <= 6 and nb_sem_av_al >= 0:
        return 3
    else:
        return 4


def details_modification(first_row, operation, row):
    """
    Handles the final decision tree branch for extensions.
    Exact replica of original function.
    """
    if first_row['capa_increase'] == 'OSS - NOK':
        return "OSS - NOK : voir pourquoi ça ne remonte pas dans les bases"
    
    if first_row['etat_last_projet'] == 'EN COURS':
        return "ne pas lancer, VDR en cours"
    
    if operation == 'SWAP':
        date_existante = False
        if not pd.isna(first_row['date_jakarta']):
            date_existante = True
            date_jakarta = first_row['date_jakarta']
            date_limit = datetime.now() - timedelta(days=60)
        
        if date_existante and first_row['etat_jakarta'] == 'Planifié' and date_jakarta >= date_limit:
            return "ne pas lancer, JAKARTA"
        return 'pas de pb'
    
    nb_dispo = first_row['Ports_dispo_ff']
    if nb_dispo != 0:
        if nb_dispo >= first_row['nb_pm_occup_sup_75']:
            return 'pas de pb'
        elif nb_dispo > first_row['rang_nro']:
            return "lancer extension + extension nro"
        else:
            row['Ports_nro_suffisant'] = 'non'
            return "lancer extension sur nro, on ne lance pas l'extension"
    else:
        row['Ports_nro_suffisant'] = 'non'
        return "lancer extension sur nro, on ne lance pas l'extension"


def arbre_dec(row, data_frame, data_annexe, decision_engine=None):
    """
    Refactored decision tree function.
    Delegates logic to DecisionTreeEngine while preserving exact behavior.
    """
    if decision_engine is None:
        decision_engine = DecisionTreeEngine()
    
    id_pm = row['id_pm']
    
    if data_frame.loc[data_frame['id_pm'] == id_pm, 'partenaire_ff'].empty:
        return row
    
    first_row = data_frame.loc[data_frame['id_pm'] == id_pm].squeeze()
    
    # Extract context variables (exact same logic as original)
    PA = first_row['partenaire_ff']
    PM = first_row['calibre_pm']
    nb_lien_nro_PM = first_row['pon_paths']
    SWAP_possible = first_row['elligible_swap']
    type_liaison = first_row['type_liaison']
    nom_sfp = first_row['nom_sfp']
    
    # Fibre en attente logic (exact replica)
    if (PM == '900' and nb_lien_nro_PM % 2 == 0):
        fibre_en_attente = True
    elif (PM == '1000' and nb_lien_nro_PM % 3 == 0):
        fibre_en_attente = True
    else:
        fibre_en_attente = False
    
    multiple_PON = first_row['multiple_PON']
    PON = first_row['splitter_c0'] * first_row['splitter_c1'] * first_row['splitter_c2']
    
    # Update non-decision fields
    row['occupation'] = first_row['occupation']
    row['id_nra'] = first_row['id_nra']
    row['type_PM'] = PM
    row['Partenaire / Zone'] = first_row['partenaire_fiabilise']
    row['prio_croissance'] = priorité(first_row)
    
    # Date comparison (exact same logic)
    before_pilot_date = datetime.now() < datetime.strptime(date_str, "%Y-%m-%d")
    
    # Build context for decision engine
    context = {
        'PON': PON,
        'PA': PA,
        'PM': PM,
        'nb_lien': nb_lien_nro_PM,
        'before_pilot_date': before_pilot_date,
        'SWAP_possible': SWAP_possible,
        'type_liaison': type_liaison,
        'nom_sfp': nom_sfp,
        'fibre_en_attente': fibre_en_attente,
        'multiple_PON': multiple_PON,
        'annexe': data_annexe['nom_sfp'].values
    }
    
    # Apply decision rules
    result = decision_engine.apply_rules(context)
    
    if result is None:
        # No rule matched - should not happen with proper rules
        row['commentaire'] = 'Pas de règle correspondante'
        return row
    
    action, operation, comment, nb_fibre, stop = result
    
    # Handle special actions
    if action == 'send_to_tafi':
        row['commentaire'] = comment
        return row
    
    if action == 'error':
        print(comment)
        row['commentaire'] = comment
        return row
    
    if action == 'no_solution':
        row['commentaire'] = comment
        return row
    
    # Handle SWAP operations
    if action == 'SWAP':
        detail = details_modification(first_row, 'SWAP', row)
        if detail != 'pas de pb':
            row['commentaire'] = detail
        else:
            row['commentaire'] = comment
            row['actions'] = operation
        return row
    
    # Handle SWAP_PLUS (special case for PON64 ASTERIX)
    if action == 'SWAP_PLUS':
        detail = details_modification(first_row, 'SWAP', row)
        if detail != 'pas de pb':
            row['commentaire'] = detail
        else:
            row['commentaire'] = comment
            row['actions'] = operation
        return row
    
    # Handle FO (fiber addition) operations
    if action == 'FO':
        detail = details_modification(first_row, 'fo', row)
        if detail in ('pas de pb', "lancer extension + extension nro"):
            action_base = comment
            if detail == "lancer extension + extension nro":
                action_base += ' + extension nro'
            row['commentaire'] = action_base
            row['actions'] = action_base
            row['nb_fibre_to_add'] += nb_fibre
        else:
            row['commentaire'] = detail
        return row
    
    return row


# Keep other functions unchanged
def format_data(data_prod_info_eb_zmd, data_croissance, data_liaison, data_sfp, 
                data_fin: pd.DataFrame, data_STG, data_annexe):
    """Original format_data function - unchanged"""
    data_NC = pd.merge(data_liaison, data_sfp, left_on=['id_olt', 'carte_olt', 'port_olt'],
                       right_on=['id_olt', 'num_slot_carte', 'num_slot_sfp'], how='left')
    
    data = pd.merge(data_prod_info_eb_zmd, data_croissance, on='id_pm', how='left')
    data = pd.merge(data, data_NC, left_on=['id_pm', 'id_nro'], right_on=['id_pm', 'id_nro'], how='left')
    data = pd.merge(data, data_STG, on='id_olt', how='left')
    
    data.set_index('id_pm', inplace=True, drop=False)
    data = add_multiple_PON_columns(data)
    data = data.drop_duplicates(subset='id_pm', keep='first')
    
    data['PDL'] = data.apply(
        lambda row: np.nan if row['lr_date'] == 0 else 100.0 * row['port_total'] / row['lr_date'],
        axis=1
    )
    
    data.sort_values(by=['occupation', 'port_libre', 'PDL'], ascending=[False, True, True], inplace=True)
    data['rang_global'] = np.arange(1, len(data) + 1)
    data['rang_nro'] = data.groupby('id_nro')['rang_global'].rank(method='min', ascending=True)
    data['nb_sem_avant_al'] = data['nb_sem_avant_al'].astype(float)
    
    print("data pret")
    
    data_fin['id_nra'] = np.nan
    data_fin['type_PM'] = np.nan
    data_fin['Partenaire / Zone'] = np.nan
    data_fin['nb_fibre_to_add'] = 0
    data_fin['prio_croissance'] = np.nan
    data_fin['actions'] = np.nan
    data_fin['Ports_nro_suffisant'] = 'oui'
    data_fin['commentaire'] = np.nan
    
    data_fin = data_fin.drop_duplicates(subset='id_pm', keep='first')
    data_fin.set_index('id_pm', inplace=True, drop=False)
    
    print("data_fin pret")
    
    return data, data_fin, data_annexe


def format_push(data_fin: pd.DataFrame):
    """Original format_push function - unchanged"""
    data_fin['PDL'] = data_fin.apply(
        lambda row: np.nan if row['lr_date'] == 0 else 100.0 * row['port_total'] / row['lr_date'],
        axis=1
    )
    
    non_null_actions = data_fin.dropna(subset=['actions'])
    null_actions = data_fin[data_fin['actions'].isnull()]
    
    non_null_actions.sort_values(
        by=['prio_croissance', 'occupation', 'port_libre', 'PDL'],
        ascending=[True, False, True, True],
        inplace=True
    )
    non_null_actions['rang_global'] = np.arange(1, len(non_null_actions) + 1)
    
    dernier_rang_non_nul = non_null_actions['rang_global'].max()
    
    null_actions.sort_values(
        by=['occupation', 'port_libre', 'PDL'],
        ascending=[False, True, True],
        inplace=True
    )
    null_actions['rang_global'] = np.arange(
        dernier_rang_non_nul + 1,
        dernier_rang_non_nul + 1 + len(null_actions)
    )
    
    data_fin = pd.concat([non_null_actions, null_actions])
    
    return data_fin


def add_multiple_PON_columns(data_frame):
    """Original add_multiple_PON_columns function - unchanged"""
    def check_multiple_PON(group):
        if len(group) > 1:
            PON = group.iloc[0]['splitter_c0'] * group.iloc[0]['splitter_c1'] * group.iloc[0]['splitter_c2']
            group['multiple_PON'] = group.apply(
                lambda row: row['splitter_c0'] * row['splitter_c1'] * row['splitter_c2'] != PON,
                axis=1
            )
        else:
            group['multiple_PON'] = False
        return group
    
    data_frame['multiple_PON'] = False
    data_frame = data_frame.groupby(level='id_pm').progress_apply(check_multiple_PON)
    return data_frame
