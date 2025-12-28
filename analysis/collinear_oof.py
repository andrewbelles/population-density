import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import cohen_kappa_score

def analyze_collinearity(path_a, path_b):
    print(f"--- Collinearity Analysis ---")
    print(f"A: {path_a}")
    print(f"B: {path_b}")

    try:
        mat_a = loadmat(path_a)
        mat_b = loadmat(path_b)
    except Exception as e:
        print(f"Error loading .mat files: {e}")
        return

    # Helper to extract data safely
    def extract_data(mat):
        # In OOF datasets, 'features' are usually the probabilities from the previous layer
        if 'features' in mat: return mat['features']
        if 'probabilities' in mat: return mat['probabilities']
        raise ValueError("Key 'features' or 'probabilities' not found.")

    probs_a = extract_data(mat_a)
    probs_b = extract_data(mat_b)
    
    # Handle FIPS formatting for alignment
    # FIPS might be stored as strings inside arrays
    try:
        fips_a = [str(x).strip().replace("['", "").replace("']", "") for x in mat_a['fips_codes'].flatten()]
        fips_b = [str(x).strip().replace("['", "").replace("']", "") for x in mat_b['fips_codes'].flatten()]
    except:
        # Fallback if fips_codes are simple numpy arrays
        fips_a = mat_a['fips_codes'].flatten()
        fips_b = mat_b['fips_codes'].flatten()

    # Create DataFrames for safe alignment
    df_a = pd.DataFrame(probs_a, index=fips_a)
    df_b = pd.DataFrame(probs_b, index=fips_b)

    # Inner Join to find common counties
    common_fips = df_a.index.intersection(df_b.index)
    
    if len(common_fips) == 0:
        print("[ERROR] No matching FIPS codes found. Check format.")
        return
        
    df_a = df_a.loc[common_fips]
    df_b = df_b.loc[common_fips]
    
    print(f"Aligned {len(common_fips)} counties.\n")

    # 1. Class-wise Correlation (Pearson R)
    correlations = []
    print(f"{'Class':<6} | {'Correlation (r)':<15} | {'Interpretation'}")
    print("-" * 45)
    
    for i in range(df_a.shape[1]):
        # Correlation between Model A's Class 0 prob and Model B's Class 0 prob
        r = np.corrcoef(df_a.iloc[:, i], df_b.iloc[:, i])[0, 1]
        correlations.append(r)
        
        interp = "Orthogonal"
        if r > 0.9: interp = "Identity (Bad)"
        elif r > 0.7: interp = "High Collinearity"
        elif r > 0.5: interp = "Moderate"
        
        print(f"{i:<6} | {r:.4f}{' '*10} | {interp}")

    avg_corr = np.mean(correlations)
    print("-" * 45)
    print(f"Average Correlation: {avg_corr:.4f}")
    
    if avg_corr > 0.75:
        print("\n[DIAGNOSIS] HIGH COLLINEARITY DETECTED.")
        print("The models are learning the exact same signal.")
    elif avg_corr < 0.5:
        print("\n[DIAGNOSIS] LOW COLLINEARITY.")
        print("The models are finding different patterns (Good).")

    # 2. Hard Prediction Agreement
    preds_a = df_a.values.argmax(axis=1)
    preds_b = df_b.values.argmax(axis=1)
    
    agreement = np.mean(preds_a == preds_b)
    kappa = cohen_kappa_score(preds_a, preds_b)
    
    print("\n--- Decision Agreement ---")
    print(f"Raw Match Rate: {agreement*100:.2f}% (Counties where A & B predict same class)")
    print(f"Cohen's Kappa:  {kappa:.4f} (0=Random, 1=Perfect Agreement)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_collinearity.py <oof_dataset_A.mat> <oof_dataset_B.mat>")
    else:
        analyze_collinearity(sys.argv[1], sys.argv[2])
