"""
Refactored decision tree with YAML rules - COMPLETE & CORRECTED
Exact behavioral equivalence with original
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import yaml
from pathlib import Path

# Global pilot date
date_str = "2025-12-01"


class DecisionTreeEngine:
    """Optimized decision tree engine with exact behavioral match"""
    
    def __init__(self, rules_path=None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "decision_rules.yaml"
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Sort by order to ensure first-match semantics
        self.global_checks = sorted(
            self.config.get('global_checks', []),
            key=lambda x: x.get('order', 0)
        )
        self.rules = sorted(
            self.config.get('rules', []),
            key=lambda x: x.get('order', 0)
        )
    
    def evaluate_condition(self, condition, context):
        """Evaluate single condition with exact operator semantics"""
        field = condition['field']
        op = condition['op']
        value = condition.get('value')
        
        field_val = context.get(field)
        
        # Handle None/NaN
        if pd.isna(field_val):
            if op == "is_null":
                return True
            elif op == "not_null":
                return False
            else:
                return False
        
        # Comparison operators
        if op == "==":
            return field_val == value
        elif op == "!=":
            return field_val != value
        elif op == "<":
            return field_val < value
        elif op == "<=":
            return field_val <= value
        elif op == ">":
            return field_val > value
        elif op == ">=":
            return field_val >= value
        elif op == "in":
            return field_val in value
        elif op == "not_in":
            return field_val not in value
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def matches_rule(self, rule, context):
        """Check if all conditions match (AND logic)"""
        return all(
            self.evaluate_condition(cond, context)
            for cond in rule['conditions']
        )
    
    def find_matching_rule(self, context):
        """Find first matching rule (first-match semantics)"""
        # Check global conditions first
        for check in self.global_checks:
            conditions = check.get('conditions', [])
            if self.matches_rule({'conditions': conditions}, context):
                return check
        
        # Check main rules in order
        for rule in self.rules:
            if self.matches_rule(rule, context):
                return rule
        
        return None


def priorité(first_row):
    """EXACT copy from original"""
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
    """EXACT copy from original"""
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
    Refactored decision tree using YAML rules.
    EXACT behavioral equivalence with original.
    """
    if decision_engine is None:
        decision_engine = DecisionTreeEngine()
    
    id_pm = row['id_pm']
    
    # Early return if no data
    if data_frame.loc[data_frame['id_pm'] == id_pm, 'partenaire_ff'].empty:
        return row
    
    first_row = data_frame.loc[data_frame['id_pm'] == id_pm].squeeze()
    
    # Extract variables EXACTLY as original
    PA = first_row['partenaire_ff']
    PM = first_row['calibre_pm']
    nb_lien_nro_PM = first_row['pon_paths']
    SWAP_possible = first_row['elligible_swap']
    type_liaison = first_row['type_liaison']
    nom_sfp = first_row['nom_sfp']
    
    # Fibre en attente logic - EXACT
    if (PM == '900' and nb_lien_nro_PM % 2 == 0):
        fibre_en_attente = True
    elif (PM == '1000' and nb_lien_nro_PM % 3 == 0):
        fibre_en_attente = True
    else:
        fibre_en_attente = False
    
    multiple_PON = first_row['multiple_PON']
    PON = first_row['splitter_c0'] * first_row['splitter_c1'] * first_row['splitter_c2']
    
    # Update non-decision fields - EXACT
    row['occupation'] = first_row['occupation']
    row['id_nra'] = first_row['id_nra']
    row['type_PM'] = PM
    row['Partenaire / Zone'] = first_row['partenaire_fiabilise']
    row['prio_croissance'] = priorité(first_row)
    
    # Date comparison - EXACT
    before_pilot_date = datetime.now() < datetime.strptime(date_str, "%Y-%m-%d")
    
    # Build context
    context = {
        'PON': PON,
        'PA': PA,
        'PM': PM,
        'nb_lien': nb_lien_nro_PM,
        'nb_lien_mod_2': nb_lien_nro_PM % 2,
        'before_pilot_date': before_pilot_date,
        'SWAP_possible': SWAP_possible,
        'type_liaison': type_liaison,
        'nom_sfp': nom_sfp,
        'fibre_en_attente': fibre_en_attente,
        'multiple_PON': multiple_PON,
        'nom_sfp_in_annexe': nom_sfp in data_annexe['nom_sfp'].values if not pd.isna(nom_sfp) else False,
    }
    
    # Find matching rule
    matched_rule = decision_engine.find_matching_rule(context)
    
    if matched_rule is None:
        row['commentaire'] = 'Pas de règle correspondante'
        return row
    
    action = matched_rule['action']
    action_type = action['type']
    
    # Handle action types
    if action_type == "comment_only":
        row['commentaire'] = action['commentaire']
        if matched_rule.get('stop', False):
            return row
    
    elif action_type == "error":
        print(action['commentaire'])
        row['commentaire'] = action['commentaire']
    
    elif action_type == "validate_swap":
        detail = details_modification(first_row, 'SWAP', row)
        if detail != 'pas de pb':
            row['commentaire'] = detail
        else:
            row['commentaire'] = action['commentaire']
            row['actions'] = action['actions']
    
    elif action_type == "validate_fo":
        detail = details_modification(first_row, 'fo', row)
        if detail in ('pas de pb', "lancer extension + extension nro"):
            action_base = action['commentaire']
            if detail == "lancer extension + extension nro":
                action_base += ' + extension nro'
            row['commentaire'] = action_base
            row['actions'] = action_base
            row['nb_fibre_to_add'] += action.get('nb_fibre', 0)
        else:
            row['commentaire'] = detail
    
    return row


def format_data(data_prod_info_eb_zmd, data_croissance, data_liaison, data_sfp,
                data_fin: pd.DataFrame, data_STG, data_annexe):
    """EXACT copy from original"""
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
    """EXACT copy from original"""
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
    """EXACT copy from original"""
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
