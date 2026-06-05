@echo off
title DDDM Decision Support System Launcher
color 0A

echo ======================================================================
echo          DDDM Decision Support System - Customer Retention & CLV
echo ======================================================================
echo.
echo Ce script va initialiser l'environnement, verifier les datasets,
echo entrainer la pipeline de Machine Learning (XGBoost) et lancer
echo l'application interactive Streamlit.
echo.
echo ----------------------------------------------------------------------
echo Etape 1 : Installation des dependances Python...
echo ----------------------------------------------------------------------
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo AVERTISSEMENT : L'installation de certaines dependances a echoue ou est deja satisfaite.
    echo Tentative de poursuite du script...
)

echo.
echo ----------------------------------------------------------------------
echo Etape 2 : Generation des profils clients (demographies)...
echo ----------------------------------------------------------------------
python data/generate_demographics.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERREUR : Echec de l'acquisition des donnees.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ----------------------------------------------------------------------
echo Etape 3 : Entrainement des modeles de prediction & SHAP Explainer...
echo ----------------------------------------------------------------------
python src/train_models.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERREUR : Echec de l'entrainement de la pipeline de Machine Learning.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ----------------------------------------------------------------------
echo Etape 4 : Lancement du Decision Dashboard (Streamlit)...
echo ----------------------------------------------------------------------
echo Le dashboard va s'ouvrir automatiquement dans votre navigateur par defaut.
echo (Vous pouvez fermer la fenetre en pressant Ctrl+C dans ce terminal)
echo.
streamlit run dashboard/app.py

pause
