Loading data...
Opening connection with vmwres-outi220
connection établie avec Prod
connection établie avec Network_configuration
connection établie avec STG_BO
connection établie avec Annexe
100%|██████████████████████████████████████████████████████████████████████████████████████████| 86111/86111 [00:40<00:00, 2121.53it/s]
data pret
data_fin pret

Comparing 8 mismatched PMs...

====================================================================================================
COMPARING: ADR-42075-CRAB
====================================================================================================

Context:
  PON: 1 × 2 × 32 = 64
  PA: RIP Axione
  PM: None
  nb_lien: 2
  type_liaison: INI
  nom_sfp: SFP C+
  elligible_swap: NON
  nom_sfp in annexe: False

🔵 ORIGINAL:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

🟢 REFACTORED:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

  Matched rule: PON64_OTHER (order 520)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: ADR_36046_PACT
====================================================================================================

Context:
  PON: 1 × 2 × 32 = 64
  PA: RIP Axione
  PM: None
  nb_lien: 3
  type_liaison: EXT
  nom_sfp: SFP C+
  elligible_swap: NON
  nom_sfp in annexe: False

🔵 ORIGINAL:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

🟢 REFACTORED:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

  Matched rule: PON64_OTHER (order 520)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: ADR_59227_DEL1
====================================================================================================

Context:
  PON: 1 × 2 × 32 = 64
  PA: RIP Axione
  PM: None
  nb_lien: 3
  type_liaison: INI
  nom_sfp: SFP MPM C+
  elligible_swap: NON
  nom_sfp in annexe: False

🔵 ORIGINAL:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

🟢 REFACTORED:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

  Matched rule: PON64_OTHER (order 520)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: FI-03118-0002
====================================================================================================

Context:
  PON: 1 × 2 × 64 = 128
  PA: RIP OF COLLECTIVITE
  PM: None
  nb_lien: 2
  type_liaison: INI
  nom_sfp: SFP MPM C+
  elligible_swap: NON
  nom_sfp in annexe: False

🔵 ORIGINAL:
  commentaire: ajout 1 fo 1v128
  actions: ajout 1 fo 1v128
  nb_fibre_to_add: 1

🟢 REFACTORED:
  commentaire: ajout 1 fo 1v128
  actions: ajout 1 fo 1v128
  nb_fibre_to_add: 1

  Matched rule: PON128 (order 600)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: FI-03190-0004
====================================================================================================

Context:
  PON: 1 × 2 × 32 = 64
  PA: ZMD AMII ASTERIX
  PM: None
  nb_lien: 2
  type_liaison: INI
  nom_sfp: SFP Combo D3
  elligible_swap: OUI
  nom_sfp in annexe: True

🔵 ORIGINAL:
  commentaire: A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64
  actions: Swap 1v64 vers 1v128 + ajout 1 tiroir 64
  nb_fibre_to_add: 0

🟢 REFACTORED:
  commentaire: A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64
  actions: Swap 1v64 vers 1v128 + ajout 1 tiroir 64
  nb_fibre_to_add: 0

  Matched rule: PON64_ASTERIX_ELIG_INI_NE1L (order 511)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: FI-18033-0016
====================================================================================================

Context:
  PON: 1 × 2 × 32 = 64
  PA: ZMD AMII ASTERIX
  PM: None
  nb_lien: 2
  type_liaison: INI
  nom_sfp: SFP Combo D3
  elligible_swap: OUI
  nom_sfp in annexe: True

🔵 ORIGINAL:
  commentaire: A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64
  actions: Swap 1v64 vers 1v128 + ajout 1 tiroir 64
  nb_fibre_to_add: 0

🟢 REFACTORED:
  commentaire: A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64
  actions: Swap 1v64 vers 1v128 + ajout 1 tiroir 64
  nb_fibre_to_add: 0

  Matched rule: PON64_ASTERIX_ELIG_INI_NE1L (order 511)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: FI-29019-005Z
====================================================================================================

Context:
  PON: 1 × 2 × 32 = 64
  PA: ZMD AMII ASTERIX
  PM: None
  nb_lien: 2
  type_liaison: INI
  nom_sfp: SFP Combo D3
  elligible_swap: OUI
  nom_sfp in annexe: True

🔵 ORIGINAL:
  commentaire: A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64
  actions: Swap 1v64 vers 1v128 + ajout 1 tiroir 64
  nb_fibre_to_add: 0

🟢 REFACTORED:
  commentaire: A lancer en SWAP de liaison 1v64 vers 1v128 et ajout de tiroir 64
  actions: Swap 1v64 vers 1v128 + ajout 1 tiroir 64
  nb_fibre_to_add: 0

  Matched rule: PON64_ASTERIX_ELIG_INI_NE1L (order 511)

⚖️  COMPARISON:
  ✓ All fields match

Press Enter for next, 'q' to quit, 's' to save diagnosis:

====================================================================================================
COMPARING: FI-14118-002G
====================================================================================================

Context:
  PON: 1 × 1 × 64 = 64
  PA: ZMD AMII ASTERIX
  PM: None
  nb_lien: 3
  type_liaison: EXT
  nom_sfp: SFP MPM C++
  elligible_swap: NON
  nom_sfp in annexe: True

🔵 ORIGINAL:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

🟢 REFACTORED:
  commentaire: ajout 1 fo 1v64
  actions: ajout 1 fo 1v64
  nb_fibre_to_add: 1

  Matched rule: PON64_ASTERIX_ELIG_NOT_INI (order 512)

⚖️  COMPARISON:
  ✓ All fields match
