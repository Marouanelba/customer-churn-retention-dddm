import os
import pandas as pd
import numpy as np

def load_raw_data(data_dir="data"):
    """
    Charge les jeux de données de transactions et démographiques des clients.
    """
    tx_path = os.path.join(data_dir, "online_retail_raw.csv")
    demographics_path = os.path.join(data_dir, "customer_demographics.csv")
    
    if not os.path.exists(tx_path) or not os.path.exists(demographics_path):
        raise FileNotFoundError("Les jeux de données bruts sont manquants. Veuillez exécuter generate_demographics.py en premier.")
        
    # Détecter le séparateur et la décimale dynamiquement pour les transactions
    with open(tx_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    decimal = ',' if sep == ';' else '.'
    
    tx_df = pd.read_csv(tx_path, sep=sep, decimal=decimal, encoding='latin-1')
    demo_df = pd.read_csv(demographics_path)
    return tx_df, demo_df

def run_data_audit(tx_df, demo_df):
    """
    Effectue un audit complet des données (complétude, cohérence, 
    fraîcheur, granularité, doublons, et informations structurelles manquantes).
    """
    audit_report = {}
    
    # 1. Nombre de lignes et dimensions
    audit_report['tx_shape'] = tx_df.shape
    audit_report['demo_shape'] = demo_df.shape
    
    # 2. Analyse des valeurs manquantes
    audit_report['tx_missing'] = tx_df.isnull().sum().to_dict()
    audit_report['tx_missing_pct'] = (tx_df.isnull().sum() / len(tx_df) * 100).to_dict()
    
    audit_report['demo_missing'] = demo_df.isnull().sum().to_dict()
    audit_report['demo_missing_pct'] = (demo_df.isnull().sum() / len(demo_df) * 100).to_dict()
    
    # 3. Analyse des doublons
    audit_report['tx_duplicates'] = int(tx_df.duplicated().sum())
    audit_report['tx_duplicates_pct'] = float(tx_df.duplicated().sum() / len(tx_df) * 100)
    
    audit_report['demo_duplicates'] = int(demo_df.duplicated().sum())
    
    # 4. Vérifications de cohérence
    # Transactions sans ID Client
    missing_cust_count = tx_df['CustomerID'].isnull().sum()
    audit_report['tx_missing_customer_id'] = int(missing_cust_count)
    audit_report['tx_missing_customer_id_pct'] = float(missing_cust_count / len(tx_df) * 100)
    
    # Vérification du chevauchement des clients
    tx_cust_ids = set(tx_df['CustomerID'].dropna().unique())
    demo_cust_ids = set(demo_df['CustomerID'].unique())
    
    overlap = tx_cust_ids.intersection(demo_cust_ids)
    only_tx = tx_cust_ids - demo_cust_ids
    only_demo = demo_cust_ids - tx_cust_ids
    
    audit_report['unique_customers_in_tx'] = len(tx_cust_ids)
    audit_report['unique_customers_in_demo'] = len(demo_cust_ids)
    audit_report['customer_id_overlap'] = len(overlap)
    audit_report['customers_only_in_tx'] = len(only_tx)
    audit_report['customers_only_in_demo'] = len(only_demo)
    
    # 5. Valeurs aberrantes et négatives dans les transactions
    negative_qty = (tx_df['Quantity'] < 0).sum()
    negative_price = (tx_df['UnitPrice'] < 0).sum()
    zero_price = (tx_df['UnitPrice'] == 0).sum()
    
    audit_report['negative_quantity_count'] = int(negative_qty)
    audit_report['negative_unit_price_count'] = int(negative_price)
    audit_report['zero_unit_price_count'] = int(zero_price)
    
    return audit_report

def get_data_dictionary():
    """
    Retourne le dictionnaire de données documenté complet pour le jeu de données client enrichi.
    """
    dictionary = [
        # Champs Démographiques Client
        {"Column": "CustomerID", "Type": "Entier / Catégoriel", "Source": "Les deux", 
         "Description": "Identifiant unique pour chaque client."},
        {"Column": "Age", "Type": "Entier", "Source": "customer_demographics.csv", 
         "Description": "Âge du client (allant de 18 à 85 ans)."},
        {"Column": "Gender", "Type": "Catégoriel", "Source": "customer_demographics.csv", 
         "Description": "Genre du client (Homme, Femme, Non-binaire, Non divulgué)."},
        {"Column": "IncomeBracket", "Type": "Catégoriel (Ordinal)", "Source": "customer_demographics.csv", 
         "Description": "Classification des revenus (Faible, Moyen, Élevé, Très Élevé)."},
        {"Column": "PreferredChannel", "Type": "Catégoriel", "Source": "customer_demographics.csv", 
         "Description": "Canal de communication préféré (Email, SMS, Notification Push, Réseaux Sociaux, Aucun)."},
        {"Column": "MembershipLevel", "Type": "Catégoriel (Ordinal)", "Source": "customer_demographics.csv", 
         "Description": "Statut du niveau de fidélité (Bronze, Argent, Or, Platine)."},
        {"Column": "TenureYears", "Type": "Flottant", "Source": "customer_demographics.csv", 
         "Description": "Nombre d'années pendant lesquelles le client a été actif avec la marque."},
        {"Column": "SatisfactionScore", "Type": "Entier (Ordinal)", "Source": "customer_demographics.csv", 
         "Description": "Score de satisfaction client de 1 (le plus bas) à 5 (le plus haut)."},
         
        # Champs agrégés de transactions (RFM et comportement transactionnel)
        {"Column": "Recency", "Type": "Entier", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Nombre de jours entre le dernier achat du client et la date de fin du jeu de données (le plus bas est le mieux)."},
        {"Column": "Frequency", "Type": "Entier", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Nombre total de factures uniques (transactions) effectuées par le client."},
        {"Column": "Monetary", "Type": "Flottant", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Valeur monétaire totale dépensée par le client (Quantité * PrixUnitaire, net des remboursements)."},
        {"Column": "AvgOrderValue", "Type": "Flottant", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Dépense moyenne par transaction (Monétaire / Fréquence)."},
        {"Column": "RefundCount", "Type": "Entier", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Nombre de remboursements / annulations effectués par le client (Numéro de facture commençant par 'C')."},
        {"Column": "RefundRatio", "Type": "Flottant", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Proportion de transactions qui étaient des remboursements (Nombre de remboursements / Fréquence)."},
        {"Column": "Country", "Type": "Catégoriel", "Source": "online_retail_raw.csv (Agrégé)", 
         "Description": "Pays principal à partir duquel le client a passé des commandes."},
         
        # Variable cible
        {"Column": "Churn", "Type": "Binaire (Entier)", "Source": "Calculé", 
         "Description": "Variable cible pour la modélisation prédictive (1 = Désabonné, aucun achat dans les 90 derniers jours ; 0 = Retenu)."}
    ]
    return pd.DataFrame(dictionary)

def enrich_data(tx_df, demo_df):
    """
    Fonction utilitaire pour fusionner les jeux de données.
    """
    # Convertir CustomerID de manière appropriée
    tx_df = tx_df.dropna(subset=['CustomerID']).copy()
    tx_df['CustomerID'] = tx_df['CustomerID'].astype(int)
    demo_df['CustomerID'] = demo_df['CustomerID'].astype(int)
    
    # Résumer les principaux attributs des transactions par client (vue intermédiaire)
    # Nous ferons un prétraitement plus détaillé dans preprocessor.py
    tx_summary = tx_df.groupby('CustomerID').agg({
        'Country': 'first'
    }).reset_index()
    
    enriched_demo = pd.merge(demo_df, tx_summary, on='CustomerID', how='left')
    return enriched_demo

if __name__ == '__main__':
    # Simple test unitaire
    try:
        from data.generate_demographics import main as dl_main
        if not os.path.exists("data/online_retail_raw.csv"):
            dl_main()
        tx, demo = load_raw_data()
        report = run_data_audit(tx, demo)
        print("Test unitaire d'Audit des Données :")
        print(f"Lignes de transactions : {report['tx_shape'][0]}")
        print(f"Nombre de doublons dans les transactions : {report['tx_duplicates']}")
        print(f"Clients dans les données démographiques : {report['unique_customers_in_demo']}")
        print("Dictionnaire de données :")
        print(get_data_dictionary().head(3))
    except Exception as e:
        print(f"Échec du test unitaire : {e}")
