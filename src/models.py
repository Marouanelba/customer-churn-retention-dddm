import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Essaie d'importer xgboost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Essaie d'importer shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def prepare_train_test_data(encoded_df):
    """
    Sépare les caractéristiques encodées et la cible en ensembles d'entraînement et de test stratifiés,
    en mettant à l'échelle les variables numériques.
    """
    # Définit les colonnes à exclure de l'entraînement
    exclude_cols = ['CustomerID', 'Country', 'IncomeBracket', 'MembershipLevel', 'Churn']
    feature_cols = [c for c in encoded_df.columns if c not in exclude_cols]
    
    X = encoded_df[feature_cols].copy()
    y = encoded_df['Churn'].copy()
    
    # Séparation stratifiée pour préserver le ratio d'attrition (churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Met à l'échelle les caractéristiques numériques continues
    continuous_features = ['Age', 'TenureYears', 'SatisfactionScore', 'Recency', 
                           'Frequency', 'Monetary', 'AvgOrderValue', 'RefundRatio']
    
    # Assure qu'elles existent dans X
    continuous_features = [f for f in continuous_features if f in X.columns]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entraîne 3 modèles de machine learning, optimise les hyperparamètres avec GridSearchCV,
    et retourne les métriques d'évaluation comparatives.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_report = []
    trained_models = {}
    
    # 1. Régression Logistique
    print("Entraînement de la Régression Logistique...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {'C': [0.01, 0.1, 1.0, 10.0]}
    lr_grid = GridSearchCV(lr, lr_params, cv=cv, scoring='f1', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    trained_models['Logistic Regression'] = best_lr
    
    # 2. Forêt Aléatoire (Random Forest)
    print("Entraînement de la Forêt Aléatoire...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    trained_models['Random Forest'] = best_rf
    
    # 3. XGBoost / Gradient Boosting
    if XGB_AVAILABLE:
        model_name = 'XGBoost'
        print("Entraînement de XGBoost...")
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model_name = 'Gradient Boosting'
        print("Entraînement du Gradient Boosting...")
        xgb = GradientBoostingClassifier(random_state=42)
        
    xgb_params = {
        'max_depth': [3, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=cv, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    trained_models[model_name] = best_xgb
    
    # Évaluation
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        # Probabilités pour ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        models_report.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
    return pd.DataFrame(models_report), trained_models

def calculate_shap_explainer(best_model, X_train, X_test):
    """
    Calcule les valeurs SHAP globales et locales pour l'explicabilité.
    Utilise le sous-échantillonnage des données d'arrière-plan pour maintenir un calcul rapide.
    """
    if not SHAP_AVAILABLE:
        print("La bibliothèque SHAP n'est pas disponible. Ignorer le calcul SHAP.")
        return None, None
        
    print("Calcul de l'explainer et des valeurs SHAP...")
    try:
        # Pour les modèles basés sur les arbres, utilisez TreeExplainer. Pour les autres, KernelExplainer ou par défaut
        model_type = type(best_model).__name__
        
        # Échantillonne le jeu de données d'arrière-plan pour l'explainer s'il est trop grand
        background = shap.kmeans(X_train, 100) if len(X_train) > 100 else X_train
        
        if 'RandomForest' in model_type or 'XGB' in model_type or 'GradientBoosting' in model_type:
            explainer = shap.TreeExplainer(best_model)
            # TreeExplainer peut échouer ou lever des exceptions avec certaines bibliothèques
            try:
                shap_values = explainer(X_test)
            except Exception:
                explainer = shap.Explainer(best_model, background)
                shap_values = explainer(X_test)
        else:
            explainer = shap.Explainer(best_model, background)
            shap_values = explainer(X_test)
            
        return explainer, shap_values
    except Exception as e:
        print(f"Erreur lors du calcul de SHAP : {e}")
        return None, None

def save_model_artifacts(artifacts, output_dir="data"):
    """
    Enregistre le modèle et les objets de mise à l'échelle dans des fichiers pickle pour les charger dans Streamlit.
    """
    os.makedirs(output_dir, exist_ok=True)
    artifacts_path = os.path.join(output_dir, "model_artifacts.pkl")
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Artefacts du modèle mis en cache avec succès dans {artifacts_path} !")

def load_model_artifacts(output_dir="data"):
    """
    Charge les artefacts du modèle.
    """
    artifacts_path = os.path.join(output_dir, "model_artifacts.pkl")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Les artefacts du modèle à {artifacts_path} sont introuvables.")
    with open(artifacts_path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # Test unitaire
    from data_loader import load_raw_data
    from preprocessor import preprocess_and_merge
    try:
        tx, demo = load_raw_data()
        raw_enriched, encoded = preprocess_and_merge(tx, demo)
        X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data(encoded)
        report, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        print("Résumé des performances du modèle :")
        print(report)
    except Exception as e:
        print(f"Échec du test unitaire : {e}")
