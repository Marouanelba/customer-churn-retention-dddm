# E-Commerce Customer Retention & Value Optimization (DDDM)

Ce projet a été réalisé dans le cadre du module **Data-Driven Decision Making (DDDM)**. Il met en œuvre un système complet d'aide à la décision basé sur la donnée pour prédire l'attrition des clients (churn), segmenter les profils d'acheteurs (RFM), et simuler l'impact financier de campagnes de rétention ciblées.

---

## 🎯 Problématique Métier & KPIs

### 1. Question Décisionnelle Centrale
**« Comment identifier avec précision nos segments de clients à forte valeur à risque d'attrition (churn) et optimiser l'allocation de notre budget marketing de rétention afin de maximiser la CLV (Customer Lifetime Value) ? »**

### 2. L'Arbre des Indicateurs (KPI Tree)
Le KPI Tree relie nos objectifs financiers stratégiques aux indicateurs comportementaux et opérationnels de notre pipeline.

```text
Objectif Stratégique (Direction Finance & Générale)
├── Chiffre d'Affaires Net (Métrique Stratégique)
│   ├── Panier Moyen (AOV) (Métrique Tactique)
│   │   └── Montant Monétaire (RFM) (Métrique Opérationnelle)
│   └── Fréquence d'Achat Mensuelle (Métrique Tactique)
│       ├── Score de Fréquence (RFM) (Métrique Opérationnelle)
│       └── Délai Inter-achats (Métrique Opérationnelle)
└── Taux de Rétention Global (Métrique Stratégique)
    ├── Valeur Vie Client (CLV) (Métrique Tactique)
    │   ├── Probabilité de Churn Prédite (XGBoost) (Métrique Opérationnelle)
    │   └── Indice de Satisfaction Client (Métrique Opérationnelle)
    └── Taux de Rachat / Réengagement (Métrique Tactique)
        ├── Score de Récence (RFM) (Métrique Opérationnelle)
        └── Canal de Rétention le plus Performant (Métrique Opérationnelle)
```

### 3. Business Case & ROI de la Démarche Data
Fidéliser un client existant est **10 fois plus économique** que d'en recruter un nouveau :
* **CAC (Coût d'Acquisition)** : $30\ €$ / client.
* **CRC (Coût de Rétention par SMS/Email ciblé)** : $3\ €$ / client.
* **CLV Moyenne par Client** : $150\ €$.

#### Simulation de ROI (Modèle XGBoost) :
* Pour $10\ 000$ clients ciblés à risque : **Coût de campagne** = $10\ 000 \times 3\ € = 30\ 000\ €$.
* Avec un **taux de réengagement de 15%** ($1\ 500$ clients sauvés) : **CA Sauvegardé** = $1\ 500 \times 150\ € = 225\ 000\ €$.
* **Bénéfice Net** = $225\ 000\ € - 30\ 000\ € = 195\ 000\ €$.
* **Retour sur Investissement (ROI)** : **$650\%$** (chaque euro investi rapporte $6,50\ €$).

---

## 🏗️ Architecture du Système Décisionnel

```text
+-----------------------+      +-----------------------+
|  UCI Transactions     |      |  Profils Clients      |
|  (online_retail_raw)  |      |  (customer_demogr.)   |
+-----------+-----------+      +-----------+-----------+
            |                              |
            +--------------+---------------+
                           | (CustomerID join)
                           v
              +-------------------------+
              | Ingestion & Audit Data  | <--- data_loader.py
              +------------+------------+
                           |
                           v
              +-------------------------+
              |   RFM & Prétraitement   | <--- preprocessor.py
              +------------+------------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
+-----------------------+     +-----------------------+
|  Clustering RFM       |     |  Modèles de Churn     | <--- models.py
|  (K-Means k=4)        |     |  (XGBoost, RF, LR)    |
+-----------+-----------+     +-----------+-----------+
            |                             |
            |                             v
            |                 +-----------------------+
            |                 |  Interprétabilité     | <--- SHAP Explainer
            |                 |  (Global & Local)     |
            |                 +-----------+-----------+
            |                             |
            +--------------+--------------+
                           |
                           v
              +-------------------------+
              |  Interactive Dashboard  | <--- app.py (Streamlit)
              |  (Direction, Marketing, |
              |   Opérations, A/B Test) |
              +-------------------------+
```

---

## 📂 Structure du Répertoire
```text
DDDM-Projet/
├── data/
│   ├── generate_demographics.py # Génération des profils démographiques CRM à partir du fichier réel
│   ├── online_retail_raw.csv  # Transactions brutes (150 000+ lignes) 
|   |        https://archive.ics.uci.edu/dataset/352/online+retail
│   └── customer_demographics.csv # Profils et enquêtes clients (39 000+ lignes)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb # Analyse exploratoire, tests statistiques, et modélisation pas-à-pas
├── src/
│   ├── data_loader.py         # Fonctions de chargement et d'audit qualité des données
│   ├── preprocessor.py        # Agrégation comportementale RFM et encodages
│   ├── models.py              # Entraînement, CV tuning, évaluation et explicabilité SHAP
│   ├── train_models.py        # Script d'exécution global de la pipeline ML
│   └── create_notebook.py     # Générateur du Jupyter notebook d'analyse
├── dashboard/
│   └── app.py                 # Application Streamlit décisionnelle (5 vues distinctes)
├── docs/
│   ├── ab_test_plan.pdf       # Protocole expérimental complet de test A/B
│   ├── data_story.pptx        # Support de présentation exécutive
│   └── recommendations.pdf    # Recommandations stratégiques et opérationnelles
├── requirements.txt           # Dépendances Python
├── run.bat                    # Script de démarrage en 1 clic pour Windows
└── README.md                  # Ce guide explicatif
```

---

## 🛠️ Instructions d'Installation & Lancement

Le projet est conçu pour être lancé de manière extrêmement simple sur Windows grâce au script de démarrage automatique `run.bat`.

### Option A : Démarrage en 1 Clic (Recommandé)
Double-cliquez simplement sur le fichier `run.bat` à la racine du projet. 
Ce script va automatiquement :
1. Installer les dépendances listées dans `requirements.txt`.
2. Lancer la pipeline de données (génération et enrichissement des datasets).
3. Entraîner les modèles de classification (Régression logistique, Random Forest, XGBoost) et sauvegarder les fichiers de cache.
4. Lancer le tableau de bord Streamlit localement dans votre navigateur.

---

### Option B : Lancement Manuel par Étape
Si vous préférez exécuter les scripts étape par étape dans votre terminal :

**1. Installation des dépendances :**
```bash
pip install -r requirements.txt
```

**2. Acquisition et audit des données :**
```bash
python data/generate_demographics.py
```

**3. Entraînement de la pipeline Machine Learning :**
```bash
python src/train_models.py
```

**4. Démarrage du dashboard interactif Streamlit :**
```bash
streamlit run dashboard/app.py
```

---

## 📈 Résultats des Tests Statistiques (EDA)
*   **Canal de contact & Churn** : Le test du **Chi-deux** valide l'existence d'une association statistiquement significative ($p < 0.05$). L'Email ciblé affiche les meilleures performances de rétention.
*   **Spend Client & Churn** : Le test de **Mann-Whitney U** démontre de manière hautement significative ($p < 0.001$) que les clients fidèles génèrent un montant d'achat médian nettement supérieur.
*   **Satisfaction & Abonnements** : L'**ANOVA** confirme des variations de satisfaction significatives entre les différents niveaux d'abonnements ($p < 0.05$).

---

## 🏆 Performances Prédictives du Churn (XGBoost)
*   **Exactitude (Accuracy)** : **89.5%**
*   **Score F1** : **85.8%** (équilibre parfait entre Précision de ciblage et Rappel des clients à risque).
*   **ROC-AUC** : **93.1%** (capacité exceptionnelle du modèle à différencier les acheteurs actifs des clients sur le point de partir).
