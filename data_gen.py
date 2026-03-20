import pandas as pd
import numpy as np

np.random.seed(42)

def generate_entity_df(n=1000, class_ratios=None):
    """
    Génère un dataframe d'entités synthétiques avec des distributions réalistes.
    Basé sur les observations de Zola et al. (Table I) et la logique métier.

    Args: 
        n (int): Le nombre d'entités que l'on souhaite créer.
        class_ratios (dict): Le pourcentage souhaitée d'entité par classe.

    Returns:
        df (pd.DataFrame): Un DataFrame contenant n entités avec des features réalistes. 
    """
    
    # Configuration des distributions des classes
    # Par défaut, on suit approximativement les proportions du dataset WalletExplorer (Zola)
    if class_ratios is None:
        class_ratios = {
            'Exchange': 0.44,
            'Gambling': 0.24,
            'Marketplace': 0.07,
            'Mining Pool': 0.08,
            'Mixer': 0.12,
            'Service': 0.05
        }
    
    # Normalisation pour s'assurer que la somme fait 1.0
    classes = list(class_ratios.keys())
    probs = list(class_ratios.values())
    probs = [p / sum(probs) for p in probs] 
    
    # Assignation des classes aux n entités
    entity_classes = np.random.choice(classes, size=n, p=probs)
    
    # Création du DataFrame vide
    df = pd.DataFrame({
        'entity_id': [f'ent_{i}' for i in range(n)],
        'label': entity_classes
    })
    
    # Initialisation des features
    features = ['amount_received', 'amount_sent', 'balance',
                'n_tx_received', 'n_tx_sent',
                'n_addr_received', 'n_addr_sent']
    
    data = {col: np.zeros(n) for col in features}
    
    # Génération des données selon la "Logique Métier" de chaque classe
    # Utilisation de np.random.lognormal pour avoir des valeurs positives avec une "longue traîne" (effet baleine)
    for cls in classes:
        mask = df['label'] == cls
        count = mask.sum()
        
        # On passe si la classe n'est pas présente (évite les erreurs sur count=0)
        if count == 0: continue
        
        if cls == 'Exchange':
            # Le géant financier. Stocke énormément (Cold Wallets).
            data['amount_received'][mask] = np.random.lognormal(10, 2, count)
            data['amount_sent'][mask] = data['amount_received'][mask] * np.random.uniform(0.9, 1.1, count)
            data['balance'][mask] = data['amount_received'][mask] * np.random.uniform(0.15, 0.30, count)
            data['n_tx_received'][mask] = np.random.lognormal(9, 1.5, count) 
            data['n_tx_sent'][mask] = np.random.lognormal(9, 1.5, count)
            data['n_addr_received'][mask] = np.random.lognormal(10, 2, count)
            data['n_addr_sent'][mask] = np.random.lognormal(8, 2, count)
            
        elif cls == 'Mining Pool':
            # L'entonnoir. Reçoit des blocs (gros), redistribue aux mineurs (petits).
            data['amount_received'][mask] = np.random.lognormal(9, 1.5, count) 
            data['amount_sent'][mask] = data['amount_received'][mask] * np.random.uniform(0.95, 1.0, count)
            data['balance'][mask] = data['amount_received'][mask] * np.random.uniform(0.01, 0.05, count)
            data['n_tx_received'][mask] = np.random.lognormal(5, 1, count) 
            data['n_tx_sent'][mask] = np.random.lognormal(9.5, 1, count)
            data['n_addr_received'][mask] = np.random.lognormal(2.5, 0.5, count) 
            data['n_addr_sent'][mask] = np.random.lognormal(5, 1, count)
            
        elif cls == 'Gambling':
            # Le casino. Flux rapides entrants/sortants.
            data['amount_received'][mask] = np.random.lognormal(6, 1.5, count)
            data['amount_sent'][mask] = data['amount_received'][mask] * np.random.uniform(0.8, 1.2, count)
            data['balance'][mask] = data['amount_received'][mask] * np.random.uniform(0.05, 0.15, count)
            data['n_tx_received'][mask] = np.random.lognormal(7.5, 1.5, count)
            data['n_tx_sent'][mask] = np.random.lognormal(7.5, 1.5, count)
            data['n_addr_received'][mask] = np.random.lognormal(6, 1.5, count)
            data['n_addr_sent'][mask] = np.random.lognormal(6, 1.5, count)
            
        elif cls == 'Mixer':
            # L'anonymiseur. Ne garde rien, fait juste transiter.
            data['amount_received'][mask] = np.random.lognormal(5, 1.5, count)
            data['amount_sent'][mask] = data['amount_received'][mask] * np.random.uniform(0.98, 1.0, count)
            data['balance'][mask] = data['amount_received'][mask] * np.random.uniform(0.001, 0.02, count)
            data['n_tx_received'][mask] = np.random.lognormal(5, 1, count)
            data['n_tx_sent'][mask] = np.random.lognormal(5, 1, count)
            data['n_addr_received'][mask] = np.random.lognormal(7, 1.5, count)
            data['n_addr_sent'][mask] = np.random.lognormal(7, 1.5, count)

        elif cls == 'Marketplace':
            # Le commerçant obscur. Escrow en attente de livraison.
            data['amount_received'][mask] = np.random.lognormal(7, 1.5, count)
            data['amount_sent'][mask] = data['amount_received'][mask] * np.random.uniform(0.7, 0.9, count)
            data['balance'][mask] = data['amount_received'][mask] * np.random.uniform(0.05, 0.20, count)
            data['n_tx_received'][mask] = np.random.lognormal(6.5, 1, count)
            data['n_tx_sent'][mask] = np.random.lognormal(4, 1, count)
            data['n_addr_received'][mask] = np.random.lognormal(6, 1.5, count)
            data['n_addr_sent'][mask] = np.random.lognormal(4, 1.5, count)
            
        elif cls == 'Service':
            # Wallet standard ou processeur de paiement.
            data['amount_received'][mask] = np.random.lognormal(4, 1, count)
            data['amount_sent'][mask] = data['amount_received'][mask] * np.random.uniform(0.8, 1.1, count)
            data['balance'][mask] = data['amount_received'][mask] * np.random.uniform(0.02, 0.10, count)
            data['n_tx_received'][mask] = np.random.lognormal(4, 1, count)
            data['n_tx_sent'][mask] = np.random.lognormal(4, 1, count)
            data['n_addr_received'][mask] = np.random.lognormal(3, 1, count)
            data['n_addr_sent'][mask] = np.random.lognormal(3, 1, count)
    
    # Assignation et conversion des types
    for col in features:
        df[col] = np.round(data[col], 2)
        if 'n_' in col:
            df[col] = df[col].astype(int)
            df[col] = df[col].clip(lower=1)
        else:
            if col == 'balance':
                df[col] = df[[col, 'amount_received']].min(axis=1)
            df[col] = df[col].clip(lower=0.0)

    return df

def distribute_amount_powerlaw_vectorized(total_amount, n_parts):
    """
    Divise un montant total en n parts selon une loi de puissance (Pareto).

    Args:
        total_amount (int or float): Somme totale à partager.
        n_parts (int): Nombre de part à faire.

    Returns:
        amounts () 
    """
    if n_parts <= 0: return np.array([])
    if n_parts == 1: return np.array([total_amount])
    
    # Génération vectorisée rapide
    # On utilise lognormal(0, 2) pour simuler la disparité extrême du Bitcoin
    weights = np.random.lognormal(0, 2, n_parts)
    
    # Normalisation vectorisée
    sum_weights = np.sum(weights)
    weights = weights / sum_weights
    
    # Distribution
    amounts = weights * total_amount
    return amounts

def generate_address_df(df_entity, sampling_ratio=0.01):
    """
    Génère le dataframe des adresses appartenant à des entités.

    Args:
        df_entity (pd.DataFrame): Le DataFrame des entités parentes.
        sampling_ratio (float): Le pourcentage des adresses d'une entité qu'on souhaite générer.
    
    Returns:
        df (pd.DataFrame): Un DataFrame contenant les adresses des entités avec des features réalistes.
    """
    all_dfs = [] 
    
    for _, row in df_entity.iterrows():
        entity_id = row['entity_id']
        label = row['label']
        
        # Nombre d'adresses à simuler
        real_n_addr = max(row['n_addr_received'], row['n_addr_sent'])
        n_rows = int(real_n_addr * sampling_ratio)
        n_rows = max(10, n_rows) 
        n_rows = min(n_rows, 500000)
        
        # Distribution des Montants (Cohérence avec l'entité)
        # Partage du total de l'entité sur ses adresses
        addrs_recv_amt = distribute_amount_powerlaw_vectorized(row['amount_received'], n_rows)
        addrs_sent_amt = distribute_amount_powerlaw_vectorized(row['amount_sent'], n_rows)
        addrs_bal = distribute_amount_powerlaw_vectorized(row['balance'], n_rows)
        
        # Distribution des Nombres de Transactions (Cohérence avec l'entité)
        addrs_n_tx_recv = distribute_amount_powerlaw_vectorized(row['n_tx_received'], n_rows).astype(int)
        addrs_n_tx_sent = distribute_amount_powerlaw_vectorized(row['n_tx_sent'], n_rows).astype(int)
        
        # Vérification qu'il y ai au moins 1 tx
        zero_activity_mask = (addrs_n_tx_recv + addrs_n_tx_sent) == 0
        addrs_n_tx_recv[zero_activity_mask] = 1

        total_tx = addrs_n_tx_recv + addrs_n_tx_sent
        uniqueness = (total_tx == 1).astype(int)
        
        # Feature: Siblings (Comportemental)
        # Dépend de la classe : Exchanges/Mixers groupent beaucoup d'inputs
        if label in ['Exchange', 'Mixer']:
            siblings = np.random.lognormal(4, 1.5, n_rows).astype(int)
        elif label == 'Mining Pool':
            siblings = np.random.lognormal(2, 1, n_rows).astype(int)
        else:
            siblings = np.random.lognormal(0.5, 0.5, n_rows).astype(int)

        # Assemblage du DataFrame
        df_chunk = pd.DataFrame({
            'address_id': [f"{entity_id}_a{i}" for i in range(n_rows)],
            'entity_id': entity_id,
            'label': label,
            'n_tx_received': addrs_n_tx_recv,
            'n_tx_sent': addrs_n_tx_sent,
            'addr_amount_received': np.round(addrs_recv_amt, 4),
            'addr_amount_sent': np.round(addrs_sent_amt, 4),
            'addr_balance': np.round(addrs_bal, 4),
            'uniqueness': uniqueness,
            'siblings': siblings
        })
        
        all_dfs.append(df_chunk)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Nettoyage final
    df['siblings'] = df['siblings'].clip(lower=0)
    
    return df

def generate_motif1_df(df_entity, sampling_ratio=0.05):
    """
    Génère le dataframe des 1-motifs pour les entités.
    Simule les interactions Entité -> Transaction -> Entité.

    Args:
        df_entity (pd.DataFrame): Le DataFrame des entités parentes.
        sampling_ratio (float): Le pourcentage de transactions à générer en motifs 
    
    Returns:
        df (pd.DataFrame): Un DataFrame contenant les 1-motifs des entités avec des features réalistes.
    """
    all_dfs = []

    for _, row in df_entity.iterrows():
        entity_id = row['entity_id']
        label = row['label']
        
        # Nombre de motifs à simuler
        # Basé sur le nombre de transactions envoyées par l'entité
        n_rows = int(row['n_tx_sent'] * sampling_ratio)
        n_rows = max(5, n_rows) 
        n_rows = min(n_rows, 100000)
        
        # Feature : Direct Loop vs Direct Distinct
        # Les Mixers et Exchanges font tourner l'argent sur eux-mêmes
        if label == 'Mixer':
            prob_loop = 0.40 
        elif label == 'Exchange':
            prob_loop = 0.20
        else:
            prob_loop = 0.02
            
        is_direct_loop = np.random.choice([1, 0], size=n_rows, p=[prob_loop, 1-prob_loop])
        
 
        # Distribution du montant envoyé
        amount_sent = distribute_amount_powerlaw_vectorized(row['amount_sent'], n_rows)
        
        # Le montant reçu dépend du type de boucle
        # Si Loop: On reçoit quasi tout ce qu'on a envoyé (moins frais)
        # Si Distinct: C'est un paiement vers autrui, le montant "reçu" par la destination varie
        amount_received = amount_sent.copy()
        
        # Pour les Distincts, on ajoute du bruit
        mask_distinct = (is_direct_loop == 0)
        amount_received[mask_distinct] = amount_sent[mask_distinct] * np.random.uniform(0.1, 1.0, size=mask_distinct.sum())
        
        # Fee 
        # Généralement une petite fraction ou une valeur fixe
        fees = np.random.lognormal(-8, 1, n_rows)
        # Vérification que les fees ne dépassent pas le montant envoyé
        fees = np.minimum(fees, amount_sent * 0.1)

        # Adresses distinctes (Complexité de la Tx)
        # Mixers : Utilisent énormément d'adresses en entrée
        if label == 'Mixer':
            n_addr_sent = np.random.lognormal(3, 1, n_rows).astype(int) + 1 
            n_addr_recv = np.random.lognormal(3, 1, n_rows).astype(int) + 1
        elif label == 'Exchange':
            n_addr_sent = np.random.lognormal(2, 1, n_rows).astype(int) + 1
            n_addr_recv = np.random.lognormal(2, 1, n_rows).astype(int) + 1
        else:
            n_addr_sent = np.random.lognormal(0.5, 0.5, n_rows).astype(int) + 1
            n_addr_recv = np.random.lognormal(0.5, 0.5, n_rows).astype(int) + 1

        # Transactions Similaires
        # Gambling & Pools : Répètent les mêmes paiements des milliers de fois
        if label in ['Gambling', 'Mining Pool']:
            n_similar_sent = np.random.lognormal(6, 2, n_rows).astype(int)
            n_similar_recv = np.random.lognormal(6, 2, n_rows).astype(int)
        else:
            n_similar_sent = np.random.lognormal(1, 1, n_rows).astype(int)
            n_similar_recv = np.random.lognormal(1, 1, n_rows).astype(int)

        # Assemblage du DataFrame
        df_chunk = pd.DataFrame({
            'motif_id': [f"{entity_id}_m{i}" for i in range(n_rows)],
            'entity_id': entity_id,
            'label': label,
            'amount_sent': np.round(amount_sent, 4),
            'amount_received': np.round(amount_received, 4),
            'fee': np.round(fees, 6),
            'n_distinct_addr_sent': n_addr_sent,
            'n_distinct_addr_received': n_addr_recv,
            'n_similar_sent_txs': n_similar_sent,
            'n_similar_received_txs': n_similar_recv,
            'is_direct_loop': is_direct_loop
        })
        
        all_dfs.append(df_chunk)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Nettoyage
    int_cols = ['n_distinct_addr_sent', 'n_distinct_addr_received', 
                'n_similar_sent_txs', 'n_similar_received_txs', 'is_direct_loop']
    df[int_cols] = df[int_cols].clip(lower=0)
    
    return df

def generate_motif2_df(df_entity, sampling_ratio=0.01):
    """
    Génère le dataframe des 2-motifs pour les entités
    Simule les interactions Entité -> Transaction -> Entité -> Transaction -> Entité.
    
    Args:
        df_entity (pd.DataFrame): Le DataFrame des entités parentes.
        sampling_ratio (float): Le pourcentage de chemins à simuler.
    
    Returns:
        df (pd.DataFrame): Un DataFrame contenant les 2-motifs des entités avec des features réalistes.
    """
    all_dfs = [] 
    
    for _, row in df_entity.iterrows():
        entity_id = row['entity_id']
        label = row['label']
        
        # Nombre de motifs à simuler
        n_rows = int(row['n_tx_sent'] * sampling_ratio)
        n_rows = max(5, n_rows) 
        n_rows = min(n_rows, 50000)
        
        # First Branch

        # Mixers : Beaucoup d'inputs pour brouiller
        if label == 'Mixer':
            n_in_1 = np.random.lognormal(3, 1, n_rows).astype(int) + 1 
        else:
            n_in_1 = np.random.lognormal(0.5, 0.5, n_rows).astype(int) + 1
        n_out_1 = np.random.lognormal(0.5, 0.5, n_rows).astype(int) + 1
        
        amount_sent_1 = distribute_amount_powerlaw_vectorized(row['amount_sent'], n_rows)
        amount_received_1 = amount_sent_1 * np.random.uniform(0.98, 1.0, n_rows)
        
        fee_1 = np.random.lognormal(-8, 1, n_rows)
        fee_1 = np.minimum(fee_1, amount_sent_1 * 0.1)

        if label in ['Gambling', 'Mining Pool']:
            n_sim_1 = np.random.lognormal(5, 2, n_rows).astype(int)
        else:
            n_sim_1 = np.random.lognormal(1, 1, n_rows).astype(int)

        prob_loop_1 = 0.30 if label == 'Mixer' else 0.05
        is_loop_1 = np.random.choice([1, 0], size=n_rows, p=[prob_loop_1, 1-prob_loop_1])
        is_distinct_1 = 1 - is_loop_1

        # Second Branch
        n_in_2 = np.random.lognormal(0.5, 0.5, n_rows).astype(int) + 1
        n_out_2 = np.random.lognormal(0.5, 0.5, n_rows).astype(int) + 1
        
        amount_sent_2 = amount_received_1 * np.random.uniform(0.1, 0.95, n_rows)
        amount_received_2 = amount_sent_2 * np.random.uniform(0.98, 1.0, n_rows)
        
        fee_2 = np.random.lognormal(-8, 1, n_rows)
        
        n_sim_2 = np.random.lognormal(1, 1, n_rows).astype(int)

        is_loop_2 = np.random.choice([1, 0], size=n_rows, p=[0.05, 0.95])
        is_distinct_2 = 1 - is_loop_2
        prob_cycle = 0.25 if label == 'Mixer' else (0.15 if label == 'Exchange' else 0.01)
        random_cycle = np.random.choice([1, 0], size=n_rows, p=[prob_cycle, 1-prob_cycle])
        is_loop_whole = np.maximum(is_loop_1, random_cycle)
        is_distinct_whole = 1 - is_loop_whole

        # Assemblage
        df_chunk = pd.DataFrame({
            'motif_id': [f"{entity_id}_m2_{i}" for i in range(n_rows)],
            'entity_id': entity_id,
            'label': label,
            
            # Branch 1
            'n_inputs_1': n_in_1,
            'n_outputs_1': n_out_1,
            'amount_sent_1': np.round(amount_sent_1, 4),
            'amount_received_1': np.round(amount_received_1, 4),
            'fee_1': np.round(fee_1, 6),
            'n_similar_txs_1': n_sim_1,
            'direct_loop_1': is_loop_1,
            'direct_distinct_1': is_distinct_1,
            
            # Branch 2
            'n_inputs_2': n_in_2,
            'n_outputs_2': n_out_2,
            'amount_sent_2': np.round(amount_sent_2, 4),
            'amount_received_2': np.round(amount_received_2, 4),
            'fee_2': np.round(fee_2, 6),
            'n_similar_txs_2': n_sim_2,
            'direct_loop_2': is_loop_2,
            'direct_distinct_2': is_distinct_2,
            
            # Whole Motif
            'direct_loop_whole': is_loop_whole,
            'direct_distinct_whole': is_distinct_whole
        })
        
        all_dfs.append(df_chunk)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    return df