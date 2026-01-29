import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Création d'un df_entity (inventé car pas de données pour l'instant)
data_entity = {
    'entity_id': [101, 102, 103, 104, 105],
    'entity_name': ['Kraken', 'BitMixer', 'SatoshiDice', 'Binance', 'SilkRoad'],
    'label': ['Exchange', 'Mixer', 'Gambling', 'Exchange', 'Marketplace']
}
df_entity = pd.DataFrame(data_entity)

# Création d'un df_address (idem)
data_address = {
    'address_id': [
        'A1', 'A2', 'A3', 'A4',       # Kraken (Exchange)
        'B1', 'B2', 'B3',             # BitMixer (Mixer)
        'C1', 'C2', 'C3',             # SatoshiDice (Gambling) 
        'D1', 'D2', 'D3', 'D4',       # Binance (Exchange)
        'E1', 'E2'                    # SilkRoad (Marketplace)
    ],
    'entity_id': [
        101, 101, 101, 101,
        102, 102, 102,
        103, 103, 103,
        104, 104, 104, 104,
        105, 105
    ],
    # Features inventés pour l'exemple
    'addr_recv_amount': np.random.rand(16) * 1000, 
    'addr_balance': np.random.rand(16) * 100,
    'is_unique': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    'n_siblings': np.random.randint(0, 20, 16)
}

df_address = pd.DataFrame(data_address)

# On attribue à chaque adresse le label de son entité
df_address = df_address.merge(df_entity[['entity_id', 'label']], on='entity_id', how='left')

# On va entraîner un classifier sur le df_entity brut pour servir de comparaison
features_baseline = ['total_balance', 'n_addresses', 'n_transactions']
X_ent = df_entity[features_baseline]
y_ent = df_entity['label']

# On fixe les paramètres de notre classifier en respectant le papier
C_entity = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)

# Sur des vraies données mettre 
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42) 


scores = cross_val_score(C_entity, X_ent, y_ent, cv=cv, scoring='accuracy')

print("=== RÉSULTATS BASELINE (C_entity) ===")
print(f"Accuracy moyenne : {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
print("C'est le score à battre avec la méthode Cascade !")
print("=======================================\n")



# On split les adresses en Set A et Set B
# On définit y (les labels)
y = df_address['label']

# On récupère les INDICES (les numéros de lignes) plutôt que les données directes
# C'est plus propre pour garder la trace de qui est qui.
indices = df_address.index
X_train_indices, X_test_indices = train_test_split(
    indices, 
    test_size=0.3,     # 30% dans le Set B
    random_state=42,   # Pour avoir toujours le même résultat
    stratify=y         # Garder les proportions
)

# Création des Dataframes Set A et Set B
df_A = df_address.loc[X_train_indices].copy()
df_B = df_address.loc[X_test_indices].copy()

print(f"Total Adresses : {len(df_address)}")
print(f"Taille Set A (Entrainement) : {len(df_A)}")
print(f"Taille Set B (Génération Features) : {len(df_B)}")
print("\nRépartition des classes dans le Set B :")
print(df_B['label'].value_counts())


# On entraîne le premier classifier C_address
# On initialise le Random Forest (paramètres issus du papier)
C_address = RandomForestClassifier(n_estimators=10, random_state=42)

# On entraîne SEULEMENT sur les features de A
features_cols = ['addr_recv_amount', 'addr_balance', 'is_unique', 'n_siblings']
C_address.fit(df_A[features_cols], df_A['label'])
print("Modèle Adresse entraîné sur le Set A.")

# Prédictions pour chaque adresse de sa classe
predictions = C_address.predict(df_B[features_cols])


# On crée un dataframe qui relie chaque prédictions à son entité
df_predis = pd.DataFrame({
    'entity_id': df_B['entity_id'].values,
    'predicted_class': predictions
})

# On utilise la formule du papier pour calculer les features pour enrichir df_entity
new_features = pd.crosstab(
    df_predis['entity_id'], 
    df_predis['predicted_class'], 
    normalize='index' 
)
new_features.columns = [f'add as {c}' for c in new_features.columns]

# Si une classe n'a jamais été prédites, on ajoute la feature avec 0.0 pour toutes les valeurs
expected_columns = [f'add as {c}' for c in C_address.classes_]
new_features = new_features.reindex(columns=expected_columns, fill_value=0.0)

# On remet entity_id comme colonne
new_features = new_features.reset_index()

# On enrichi le df_entity avec les nouvelles features calculées
df_final = df_entity.merge(new_features, on='entity_id', how='left')

# Si une entité n'avait aucune adresse dans le Set B (improbable mais peu arrivé sur faux jeu de données),
# on remplace tous les nan par 0.0 
new_feature_names = [col for col in new_features.columns if col != 'entity_id']
df_final[new_feature_names] = df_final[new_feature_names].fillna(0.0)


print(df_final.head())
