import os
import urllib.request
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def download_dataset(url, dest_path):
    """Downloads the dataset from a URL if it does not exist locally."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved original dataset to {dest_path}")
    else:
        print(f"Original dataset already exists at {dest_path}")

def corrupt_data(input_path, output_path):
    """Injects intentional flaws into the clean dataset to simulate real-world messy data."""
    df = pd.read_csv(input_path)
    n_rows = len(df)
    
    # 1. Create Patient IDs (PT-10001, PT-10002, etc.)
    df.insert(0, 'Patient_ID', [f"PT-{10000 + i}" for i in range(n_rows)])
    
    # 2. Convert 'sex' from 0/1 to messy strings with typos and casing variations
    # Original: 1 = male, 0 = female
    gender_map = {1: 'Male', 0: 'Female'}
    genders = df['sex'].map(gender_map).tolist()
    
    # Inject typos and variations in approx 15% of records
    typo_genders = ['M', 'F', 'male', 'female', 'Mlae', 'femal', 'FEMALE', 'MALE', 'f', 'm']
    for idx in np.random.choice(n_rows, size=int(n_rows * 0.15), replace=False):
        # Pick a random typo gender that matches the target gender type
        orig = genders[idx]
        if orig == 'Male':
            genders[idx] = np.random.choice(['M', 'male', 'Mlae', 'MALE', 'm'])
        else:
            genders[idx] = np.random.choice(['F', 'female', 'femal', 'FEMALE', 'f'])
            
    df['sex'] = genders
    df.rename(columns={'sex': 'Gender'}, inplace=True)
    
    # 3. Convert 'cp' (chest pain type: 0, 1, 2, 3) to messy text strings
    # Original categories: 0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic
    cp_map = {
        0: 'Typical Angina',
        1: 'Atypical Angina',
        2: 'Non-Anginal Pain',
        3: 'Asymptomatic'
    }
    cp_list = df['cp'].map(cp_map).tolist()
    
    # Inject typos/case differences in chest pain text
    cp_typos = {
        'Typical Angina': ['typical angina', 'TYPICAL ANGINA', 'typical-angina', 'Typ. Angina'],
        'Atypical Angina': ['atypical angina', 'atypical-angina', 'Atyp. Angina'],
        'Non-Anginal Pain': ['non-anginal pain', 'non_anginal_pain', 'Non-Anginal'],
        'Asymptomatic': ['asymptomatic', 'ASYMPTOMATIC', 'asymptomatic_']
    }
    
    for idx in np.random.choice(n_rows, size=int(n_rows * 0.12), replace=False):
        orig = cp_list[idx]
        cp_list[idx] = np.random.choice(cp_typos[orig])
        
    df['cp'] = cp_list
    df.rename(columns={'cp': 'Chest_Pain_Type'}, inplace=True)
    
    # 4. Inject Missing Values (NaNs)
    # Resting blood pressure ('trestbps') - 5% missing
    df.loc[df.sample(frac=0.05).index, 'trestbps'] = np.nan
    # Cholesterol ('chol') - 8% missing
    df.loc[df.sample(frac=0.08).index, 'chol'] = np.nan
    # Age ('age') - 3% missing
    df.loc[df.sample(frac=0.03).index, 'age'] = np.nan
    # Thalassemia ('thal') - 4% missing
    df.loc[df.sample(frac=0.04).index, 'thal'] = np.nan
    
    # 5. Inject Outliers
    # Age: add some negative values and extremely high values
    age_outliers = np.random.choice(n_rows, size=5, replace=False)
    df.loc[age_outliers[0], 'age'] = -10
    df.loc[age_outliers[1], 'age'] = 150
    df.loc[age_outliers[2], 'age'] = -2
    df.loc[age_outliers[3], 'age'] = 135
    df.loc[age_outliers[4], 'age'] = 0
    
    # Resting blood pressure (trestbps): negative values and extreme high values
    bp_outliers = np.random.choice(df[df['trestbps'].notna()].index, size=6, replace=False)
    df.loc[bp_outliers[0], 'trestbps'] = -90
    df.loc[bp_outliers[1], 'trestbps'] = 300
    df.loc[bp_outliers[2], 'trestbps'] = -120
    df.loc[bp_outliers[3], 'trestbps'] = 0
    df.loc[bp_outliers[4], 'trestbps'] = 250
    df.loc[bp_outliers[5], 'trestbps'] = 290
    
    # Cholesterol (chol): negative values and extreme high values
    chol_outliers = np.random.choice(df[df['chol'].notna()].index, size=6, replace=False)
    df.loc[chol_outliers[0], 'chol'] = -5
    df.loc[chol_outliers[1], 'chol'] = 999
    df.loc[chol_outliers[2], 'chol'] = 800
    df.loc[chol_outliers[3], 'chol'] = 0
    df.loc[chol_outliers[4], 'chol'] = -50
    df.loc[chol_outliers[5], 'chol'] = 750
    
    # 6. Inject Duplicates (approx 20 exact duplicates and 10 conflicting duplicates)
    # Exact duplicates
    exact_dup_indices = np.random.choice(n_rows, size=20, replace=True)
    exact_dups = df.iloc[exact_dup_indices].copy()
    
    # Conflicting duplicates: Same Patient_ID but different resting BP / cholesterol or disease target
    conflict_dup_indices = np.random.choice(n_rows, size=10, replace=False)
    conflict_dups = df.iloc[conflict_dup_indices].copy()
    conflict_dups['chol'] = conflict_dups['chol'] * 1.5 + np.random.randint(10, 50, size=len(conflict_dups))
    conflict_dups['trestbps'] = conflict_dups['trestbps'] + np.random.randint(10, 30, size=len(conflict_dups))
    # Toggle target in some conflicts
    conflict_dups['target'] = 1 - conflict_dups['target']
    
    # Combine everything
    df_messy = pd.concat([df, exact_dups, conflict_dups], ignore_index=True)
    
    # Shuffle the final messy dataset
    df_messy = df_messy.sample(frac=1.0).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_messy.to_csv(output_path, index=False)
    print(f"Messy data generated successfully! Size: {len(df_messy)} rows. Saved to {output_path}")

def generate_raw_data(output_path, clean_cache_path=None):
    """Main execution point for generator."""
    if clean_cache_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        clean_cache_path = os.path.join(script_dir, '..', 'data', 'heart_disease_clean_original.csv')
    
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    download_dataset(url, clean_cache_path)
    corrupt_data(clean_cache_path, output_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_out = os.path.join(script_dir, '..', 'data', 'raw_heart_disease.csv')
    generate_raw_data(raw_out)
