import pandas as pd
from scipy.stats import ks_2samp
import json
import numpy as np

def detect_drift(reference_file, production_file, threshold=0.05):
    """
    Detecte le drift entre donnees de reference et production
    """
    ref_data = pd.read_csv(reference_file)
    prod_data = pd.read_csv(production_file)
    
    drift_results = {}
    
    for column in ref_data.columns:
        if column in prod_data.columns and column != 'Exited':
            # Test de Kolmogorov-Smirnov
            statistic, p_value = ks_2samp(
                ref_data[column].dropna(),
                prod_data[column].dropna()
            )
            
            drift_detected = p_value < threshold
            
            # Convertir numpy types en Python natives pour JSON
            drift_results[column] = {
                'p_value': float(p_value),
                'statistic': float(statistic),
                'drift_detected': bool(drift_detected)  # CORRECTION ICI
            }
    
    # Rapport
    drifted_features = [f for f, r in drift_results.items() if r['drift_detected']]
    
    print("="*50)
    print("DATA DRIFT DETECTION REPORT")
    print("="*50)
    print(f"Threshold: {threshold}")
    print(f"Features analyzed: {len(drift_results)}")
    print(f"Features with drift: {len(drifted_features)}")
    print("\nDrifted features:")
    for feature in drifted_features:
        print(f"  - {feature}: p-value = {drift_results[feature]['p_value']:.4f}")
    print("="*50)
    
    return drift_results

if __name__ == "__main__":
    # Fichiers de données
    reference_file = "data/bank_churn.csv"
    production_file = "data/production_data.csv"
    
    # Vérifier si les fichiers existent
    try:
        pd.read_csv(reference_file)
        pd.read_csv(production_file)
    except FileNotFoundError as e:
        print(f"❌ Fichier manquant: {e}")
        print("Création d'un fichier de production de test...")
        
        # Créer des données de production factices
        ref_data = pd.read_csv(reference_file)
        prod_data = ref_data.copy()
        
        # Ajouter un peu de bruit pour simuler du drift
        np.random.seed(42)
        for col in ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']:
            if col in prod_data.columns:
                prod_data[col] = prod_data[col] + np.random.normal(0, prod_data[col].std() * 0.1, len(prod_data))
        
        prod_data.to_csv(production_file, index=False)
        print(f"✅ Fichier de test créé: {production_file}")
    
    # Exécuter la détection
    results = detect_drift(reference_file, production_file)
    
    # Sauvegarder les resultats
    with open("drift_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)  # CORRECTION ICI avec default=str
    
    print("\n✅ Rapport sauvegarde dans drift_report.json")
    
    # Afficher un résumé
    print("\n📊 RÉSUMÉ DU DRIFT DETECTÉ :")
    for feature, stats in results.items():
        status = "🚨 DRIFT" if stats['drift_detected'] else "✅ OK"
        print(f"{status} - {feature}: p={stats['p_value']:.4f}")