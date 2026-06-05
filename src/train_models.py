import os
import sys
import pandas as pd
import numpy as np

# S'assure que le dossier racine est dans le chemin python pour s'exécuter directement
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_data, run_data_audit
from src.preprocessor import preprocess_and_merge
from src.models import prepare_train_test_data, train_and_evaluate_models, calculate_shap_explainer, save_model_artifacts

def main():
    print("==================================================")
    # 1. Chargement des données
    print("[1/5] Chargement des fichiers de données brutes...")
    try:
        tx_df, demo_df = load_raw_data("data")
        print(f"Transactions ({len(tx_df)} lignes) et données démographiques ({len(demo_df)} lignes) chargées avec succès.")
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        print("Veuillez exécuter data/generate_demographics.py en premier pour générer le jeu de données démographiques.")
        sys.exit(1)
        
    # 2. Audit des données
    print("[2/5] Exécution de l'audit de qualité des données...")
    audit_report = run_data_audit(tx_df, demo_df)
    print(f"Audit terminé ! Transactions en double : {audit_report['tx_duplicates']}.")
    
    # 3. Prétraitement et fusion
    print("[3/5] Nettoyage et agrégation RFM des clients...")
    raw_enriched, encoded = preprocess_and_merge(tx_df, demo_df)
    print(f"Profils clients agrégés : {len(raw_enriched)} enregistrements.")
    print(f"Taux d'attrition (Churn) généré : {raw_enriched['Churn'].mean() * 100:.2f}%")
    
    # 4. Modélisation
    print("[4/5] Entraînement des classifieurs Régression Logistique, Random Forest, et XGBoost...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data(encoded)
    report, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\nRapport de performance de l'entraînement des modèles :")
    print("--------------------------------------------------")
    print(report.to_string(index=False))
    print("--------------------------------------------------")
    
    # Sélectionne le meilleur modèle en fonction du score F1
    best_model_row = report.sort_values(by='F1-Score', ascending=False).iloc[0]
    best_model_name = best_model_row['Model']
    best_model = models[best_model_name]
    print(f"Meilleur modèle performant : {best_model_name} (F1: {best_model_row['F1-Score']:.4f}, AUC: {best_model_row['ROC-AUC']:.4f})")
    
    # 5. Explications et mise en cache
    print("[5/5] Calcul de l'explicabilité du modèle et compilation des artefacts de cache...")
    explainer, shap_values = calculate_shap_explainer(best_model, X_train, X_test)
    
    # Extraction des clusters RFM à l'aide d'acheteurs actifs
    active_clients = raw_enriched[raw_enriched['Frequency'] > 0].copy()
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    scaler_rfm = StandardScaler()
    rfm_scaled = scaler_rfm.fit_transform(active_clients[rfm_cols])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    active_clients['Cluster'] = kmeans.fit(rfm_scaled).labels_
    
    cluster_profiles = active_clients.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'SatisfactionScore': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'ClientCount'}).reset_index()
    
    # Compilation des artefacts
    artifacts = {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'models_report': report,
        'explainer_serialized': explainer,
        'shap_values_serialized': shap_values,
        'raw_enriched_data': raw_enriched,
        'encoded_data': encoded,
        'cluster_profiles': cluster_profiles,
        'active_clients_with_clusters': active_clients
    }
    
    # Sauvegarde
    save_model_artifacts(artifacts, output_dir="data")
    print("==================================================")
    print("Pipeline de données et de modèles du projet préparé avec succès !")
    print("Vous pouvez maintenant exécuter 'streamlit run dashboard/app.py' pour lancer le tableau de bord.")

if __name__ == '__main__':
    main()
