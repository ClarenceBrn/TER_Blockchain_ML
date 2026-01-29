import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

def evaluate_classifier(clf, X, y, n_splits=5, return_detailed_report=False, random_state=42):
    """
    Évalue un classifieur (metrics = accuracy && MCC) via une Validation Croisée Stratifiée (Stratified K-Fold).
    
    Args:
        clf: Le classifieur scikit-learn à évaluer.
        X: Les features (pd.DataFrame).
        y: Les labels (pd.Series).
        n_splits (int): Nombre de plis pour la validation croisée.
        return_detailed_report (bool): Si True, calcule aussi le rapport complet par classe (plus lent).
        random_state (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        (dict): Dictionnaire avec les moyennes et std pour Accuracy et MCC (+ rapport complet optionnel).
    """
    # Configuration de la validation croisée
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) 

    # Définition des métriques à calculer en simultané
    scoring = {
        'accuracy': 'accuracy',
        'mcc': make_scorer(matthews_corrcoef)
    }

    # Calcul des scores de précision et MCC par Validation Croisée Stratifiée
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    
    acc_mean = scores['test_accuracy'].mean()
    acc_std = scores['test_accuracy'].std()
    
    mcc_mean = scores['test_mcc'].mean()
    mcc_std = scores['test_mcc'].std()

    print(f"Accuracy moyenne : {acc_mean:.2f} (+/- {acc_std * 2:.2f})")
    print(f"MCC moyenne : {mcc_mean:.2f} (+/- {mcc_std * 2:.2f})")
    results = {
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'mcc_mean': mcc_mean, 'mcc_std': mcc_std
    }

    # Attention : Double le temps de calcul car ré-entraîne les modèles
    if return_detailed_report:
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        report = classification_report(y, y_pred, output_dict=True)
        results['detailed_report'] = report
    
    return results

def prepare_and_init_model(df, ignore_cols, model_type='GradientBoosting', target_col='label', return_features=False, random_state=42):
    """
    Prépare les matrices X, y et initialise le classifieur choisi.
    
    Args:
        df (pd.DataFrame): Le dataframe contenant features et target.
        model_type (str): Type de modèle ('GradientBoosting', 'RandomForest', 'AdaBoost').
        ignore_cols (list): Colonnes à exclure des features.
        target_col (str): Nom de la colonne cible (défaut: 'label').
        random_state (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        clf: Le modèle initialisé (non entraîné).
        X: Le DataFrame des features.
        y: La Series des labels.
        features (list, optional): La liste des colonnes utilisées. Retourné uniquement si return_features=True.
    """

    # Retrait des colonnes qu'on ne souhaite pas utiliser comme features
    features = [c for c in df.columns if c != target_col and c not in ignore_cols]
    
    X = df[features]
    y = df[target_col]
    
    # Initialisation du modèle
    # Paramètres du papier (Section V-B)
    if model_type == 'GradientBoosting':
        clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        )
        
    elif model_type == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, random_state=random_state)
        
    elif model_type == 'AdaBoost':
        clf = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=random_state)
        
    else:
        raise ValueError(f"Modèle '{model_type}' non reconnu. Choisir : GradientBoosting, RandomForest, AdaBoost.")
    
    print(f"Préparation du modèle : {model_type}")
    print(f"Features sélectionnées : {features}")
    
    if return_features:
        return clf, X, y, features
    return clf, X, y

def run_cascade_layer(df, prefix, ignore_cols, model_type='RandomForest', test_size=0.3, target_col='label', random_state=42):
    """
    Exécute une couche complète de la cascade.
    
    Étapes :
    1. Split du DataFrame en Set A (Train) et Set B (Création des features)
    2. Entraînement du classifieur sur le Set A
    3. Prédiction des classes sur le Set B
    4. Création des nouvelles features agrégées (proportions par entité)
    
    Args:
        df (pd.DataFrame): Le DataFrame source.
        prefix (str): Le préfixe pour nommer les colonnes.
        ignore_cols (list): Colonnes à ne pas utiliser comme features.
        test_size (float): Proportion du dataset utilisé pour générer les features.
        random_state (int): Graine aléatoire pour la reproductibilité.
    
    Returns:
        pd.DataFrame: Le DataFrame des méta-features prêt à être merge.
    """
    
    # Split Stratifié (Set A vs Set B)
    # Utilisation des indices pour préserver l'intégrité des liens 'entity_id' lors de la reconstruction des Dataframes.
    X_train_idx, X_test_idx = train_test_split(
        df.index, 
        test_size=test_size, 
        stratify=df[target_col], 
        random_state=random_state
    )
    
    # Création des Dataframes Set A et Set B
    df_A = df.loc[X_train_idx].copy()
    df_B = df.loc[X_test_idx].copy()
    
    print(f"Cascade Layer : '{prefix}'")
    print(f"Split : {len(df_A)} train (Set A) / {len(df_B)} génération des features (Set B)")

    # Initialisation & Entraînement (Sur Set A)
    # Initialisation du Classifier (paramètres issus du papier)
    clf, X_A, y_A, features_used = prepare_and_init_model(
        df_A, 
        model_type=model_type, 
        ignore_cols=ignore_cols, 
        target_col=target_col,
        return_features=True,
        random_state=random_state
    )
    
    # Entraînement sur les features du Set A
    clf.fit(X_A, y_A)
    
    # Prédiction (Sur Set B)
    # Filtrage de df_B pour ne garder que les features apprises par le modèle
    X_B = df_B[features_used]

    # Prédiction de la classe pour chaque adresse
    predictions = clf.predict(X_B)

    # Calcul et affichage de la précision du modèle qui servira pour la cascade
    acc_check = accuracy_score(df_B['label'], predictions)
    print(f"Accuracy intermédiaire sur Set B : {acc_check:.2%}")
    
    # Agrégation et Création des Features (Formule du Papier)
    # Création d'un dataframe qui relie chaque prédiction à son entité
    df_predis = pd.DataFrame({
        'entity_id': df_B['entity_id'].values,
        'predicted_class': predictions
    })
    
    # Agrégation des prédictions par entité selon la formule de la section IV du papier
    new_features = pd.crosstab(
        df_predis['entity_id'], 
        df_predis['predicted_class'], 
        normalize='index' 
    )
    
    # Formatage Final
    # Renommage dynamique : 'add as Exchange', 'motif1 as Mixer'...
    new_features.columns = [f'{prefix} as {c}' for c in new_features.columns]
    
    # Alignement des colonnes : force la présence de toutes les classes possibles (via reindex)
    # pour garantir la cohérence dimensionnelle
    expected_cols = [f'{prefix} as {c}' for c in clf.classes_]
    new_features = new_features.reindex(columns=expected_cols, fill_value=0.0)
    
    # entity_id redevient une colonne
    return new_features.reset_index()