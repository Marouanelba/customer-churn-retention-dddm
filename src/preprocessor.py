import os
import pandas as pd
import numpy as np

def parse_dates(df, date_col='InvoiceDate'):
    """
    Analyse en toute sécurité la colonne datetime.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df

def calculate_rfm(tx_df):
    """
    Agrège les données au niveau des transactions en caractéristiques RFM au niveau du client.
    Gère les retours et calcule les taux de remboursement.
    """
    tx_df = tx_df.copy()
    
    # Convertit en toute sécurité InvoiceDate en datetime
    tx_df['InvoiceDate'] = pd.to_datetime(tx_df['InvoiceDate'], errors='coerce')
    tx_df = tx_df.dropna(subset=['InvoiceDate', 'CustomerID'])
    tx_df['CustomerID'] = tx_df['CustomerID'].astype(int)
    
    # Calcule TotalSpend par ligne
    tx_df['TotalSpend'] = tx_df['Quantity'] * tx_df['UnitPrice']
    
    # La date de référence est la date maximale dans les transactions + 1 jour
    ref_date = tx_df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Sépare les transactions normales et les retours
    # Les factures commençant par 'C' sont des annulations/remboursements
    tx_df['IsRefund'] = tx_df['InvoiceNo'].astype(str).str.startswith('C', na=False)
    
    # Groupe par CustomerID
    customer_groups = tx_df.groupby('CustomerID')
    
    rfm_data = []
    for cid, group in customer_groups:
        normal_tx = group[~group['IsRefund']]
        refund_tx = group[group['IsRefund']]
        
        # Récence : jours depuis la dernière transaction normale. Sinon, examine toutes les transactions.
        if len(normal_tx) > 0:
            last_purchase = normal_tx['InvoiceDate'].max()
        else:
            last_purchase = group['InvoiceDate'].max()
        recency = (ref_date - last_purchase).days
        
        # Fréquence : nombre de factures normales uniques
        frequency = normal_tx['InvoiceNo'].nunique()
        
        # Monétaire : Somme des dépenses totales (normales et remboursements combinés)
        monetary = group['TotalSpend'].sum()
        
        # Statistiques de remboursement
        refund_count = refund_tx['InvoiceNo'].nunique()
        refund_ratio = refund_count / frequency if frequency > 0 else 0.0
        
        # Pays principal
        country = group['Country'].iloc[0] if len(group) > 0 else 'Inconnu'
        
        rfm_data.append({
            'CustomerID': cid,
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'RefundCount': refund_count,
            'RefundRatio': min(1.0, refund_ratio),
            'Country': country
        })
        
    return pd.DataFrame(rfm_data)

def preprocess_and_merge(tx_df, demo_df):
    """
    Enrichit les données démographiques des clients avec des indicateurs RFM, définit la cible d'attrition (Churn),
    encode les variables et construit le jeu de données analytique final.
    """
    # 1. RFM Agrégé
    rfm_df = calculate_rfm(tx_df)
    
    # Assure que le type de CustomerID correspond
    demo_df = demo_df.copy()
    demo_df['CustomerID'] = demo_df['CustomerID'].astype(int)
    rfm_df['CustomerID'] = rfm_df['CustomerID'].astype(int)
    
    # 2. Fusionne les données démographiques et RFM
    merged_df = pd.merge(demo_df, rfm_df, on='CustomerID', how='left')
    
    # 3. Impute les métriques comportementales pour les clients sans transactions (prospects/non-acheteurs)
    # Ceux-ci représentent des prospects dans la base de données qui n'ont pas encore acheté (ou inactifs depuis longtemps)
    merged_df['Recency'] = merged_df['Recency'].fillna(365.0) # il y a 1 an
    merged_df['Frequency'] = merged_df['Frequency'].fillna(0.0)
    merged_df['Monetary'] = merged_df['Monetary'].fillna(0.0)
    merged_df['RefundCount'] = merged_df['RefundCount'].fillna(0.0)
    merged_df['RefundRatio'] = merged_df['RefundRatio'].fillna(0.0)
    merged_df['Country'] = merged_df['Country'].fillna('United Kingdom')
    
    # 4. Ingénierie des caractéristiques
    merged_df['AvgOrderValue'] = np.where(
        merged_df['Frequency'] > 0,
        merged_df['Monetary'] / merged_df['Frequency'],
        0.0
    )
    # Nous pourrions plafonner les valeurs monétaires négatives extrêmes pour éviter de fausser les modèles ML
    
    # 5. Définit la cible : Churn
    # Churn = 1 si Récence > 90 jours (aucun achat dans les 3 derniers mois) ou si Fréquence est 0, sinon 0
    merged_df['Churn'] = np.where(
        (merged_df['Recency'] > 90) | (merged_df['Frequency'] == 0),
        1,
        0
    )
    
    # 6. Encodage catégoriel (Ordinal & Nominal)
    encoded_df = merged_df.copy()
    
    # Mappage ordinal
    income_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
    membership_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    
    encoded_df['IncomeBracket_Encoded'] = encoded_df['IncomeBracket'].map(income_map).fillna(1)
    encoded_df['MembershipLevel_Encoded'] = encoded_df['MembershipLevel'].map(membership_map).fillna(0)
    
    # Regroupe les pays en binaire : Royaume-Uni vs. Autre
    encoded_df['Is_UK'] = np.where(encoded_df['Country'] == 'United Kingdom', 1, 0)
    
    # Encodage One-hot des variables nominales
    encoded_df = pd.get_dummies(encoded_df, columns=['Gender', 'PreferredChannel'], drop_first=True, dtype=int)
    
    return merged_df, encoded_df

if __name__ == '__main__':
    from data_loader import load_raw_data
    try:
        tx, demo = load_raw_data()
        raw_enriched, encoded = preprocess_and_merge(tx, demo)
        print("Test unitaire du prétraitement :")
        print(f"Dimensions enrichies : {raw_enriched.shape}")
        print(f"Dimensions encodées : {encoded.shape}")
        print(f"Taux d'attrition : {raw_enriched['Churn'].mean() * 100:.2f}%")
        print(f"Nombre de clients VIP (Or/Platine) : {raw_enriched['MembershipLevel'].isin(['Gold', 'Platinum']).sum()}")
    except Exception as e:
        print(f"Échec du test unitaire : {e}")
