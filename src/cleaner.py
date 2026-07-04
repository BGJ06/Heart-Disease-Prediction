import os
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
    def clean(self):
        """Cleans and standardizes the messy raw dataset."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found at {self.input_path}")
            
        df = pd.read_csv(self.input_path)
        stats = {}
        
        # Record initial shape
        stats['raw_rows'] = len(df)
        stats['raw_cols'] = len(df.columns)
        stats['raw_nulls'] = int(df.isna().sum().sum())
        
        # 1. Handle Duplicates
        # Count exact duplicates
        exact_dups = df.duplicated().sum()
        stats['exact_duplicates_removed'] = int(exact_dups)
        df.drop_duplicates(inplace=True)
        
        # Handle conflicting Patient IDs (same ID, different clinical attributes)
        # We find duplicate Patient_IDs
        dup_ids_count = df['Patient_ID'].duplicated().sum()
        stats['conflicting_ids_resolved'] = int(dup_ids_count)
        # Keep the first patient entry and drop the subsequent conflicting entries
        df.drop_duplicates(subset=['Patient_ID'], keep='first', inplace=True)
        
        # 2. Standardize Gender text
        # Mapping variations to Standard 'Male' and 'Female'
        def clean_gender(val):
            if pd.isna(val):
                return val
            val_str = str(val).strip().lower()
            if val_str in ['male', 'm', 'mlae']:
                return 'Male'
            elif val_str in ['female', 'f', 'femal']:
                return 'Female'
            return 'Male' # Fallback default
            
        df['Gender'] = df['Gender'].apply(clean_gender)
        
        # 3. Standardize Chest Pain Type
        def clean_cp(val):
            if pd.isna(val):
                return val
            val_str = str(val).strip().lower().replace('_', ' ').replace('-', ' ')
            if 'typical' in val_str:
                return 'Typical Angina'
            elif 'atypical' in val_str:
                return 'Atypical Angina'
            elif 'non' in val_str or 'pain' in val_str:
                return 'Non-Anginal Pain'
            elif 'asymp' in val_str:
                return 'Asymptomatic'
            return 'Typical Angina' # Fallback default
            
        df['Chest_Pain_Type'] = df['Chest_Pain_Type'].apply(clean_cp)
        
        # 4. Handle Outliers (Replace out-of-bounds with NaNs first, so we impute them later)
        # Age outliers: replace <= 0 or >= 110 with NaN
        age_outliers = df[(df['age'] <= 0) | (df['age'] >= 110)].index
        stats['age_outliers_imputed'] = len(age_outliers)
        df.loc[age_outliers, 'age'] = np.nan
        
        # Blood pressure outliers: replace <= 50 or >= 250 with NaN
        bp_outliers = df[(df['trestbps'] <= 50) | (df['trestbps'] >= 250)].index
        stats['bp_outliers_imputed'] = len(bp_outliers)
        df.loc[bp_outliers, 'trestbps'] = np.nan
        
        # Cholesterol outliers: replace <= 80 or >= 500 with NaN
        chol_outliers = df[(df['chol'] <= 80) | (df['chol'] >= 500)].index
        stats['chol_outliers_imputed'] = len(chol_outliers)
        df.loc[chol_outliers, 'chol'] = np.nan
        
        # 5. Imputation of Missing Values (and outliers that are now NaN)
        # Impute Age with overall median
        median_age = df['age'].median()
        if pd.isna(median_age) or np.isnan(median_age):
            median_age = 54.0 # UCI historical median
        df['age'] = df['age'].fillna(median_age)
        
        # Engineer Age_Group for group-wise imputation
        def get_age_group(age):
            if age <= 35: return '18-35'
            elif age <= 50: return '36-50'
            elif age <= 65: return '51-65'
            return '65+'
            
        df['Age_Group'] = df['age'].apply(get_age_group)
        
        # Impute Resting BP ('trestbps') with median grouped by Age_Group
        bp_group_medians = df.groupby('Age_Group')['trestbps'].median()
        # Fill group medians if any group is entirely NaN
        bp_group_medians = bp_group_medians.fillna(df['trestbps'].median()).fillna(130.0)
        
        def impute_bp(row):
            if pd.isna(row['trestbps']):
                return bp_group_medians[row['Age_Group']]
            return row['trestbps']
        df['trestbps'] = df.apply(impute_bp, axis=1)
        
        # Impute Cholesterol ('chol') with median grouped by Gender
        chol_gender_medians = df.groupby('Gender')['chol'].median()
        chol_gender_medians = chol_gender_medians.fillna(df['chol'].median()).fillna(240.0)
        
        def impute_chol(row):
            if pd.isna(row['chol']):
                return chol_gender_medians[row['Gender']]
            return row['chol']
        df['chol'] = df.apply(impute_chol, axis=1)
        
        # Impute Thalassemia ('thal') with the mode
        thal_mode = df['thal'].mode().iloc[0] if not df['thal'].dropna().empty else 2.0
        df['thal'] = df['thal'].fillna(thal_mode)
        
        # Impute any remaining NaNs in other columns (like ca, etc.)
        for col in df.columns:
            if col not in ['Patient_ID', 'Gender', 'Chest_Pain_Type', 'Age_Group']:
                col_mode = df[col].mode()
                if not col_mode.empty:
                    df[col] = df[col].fillna(col_mode.iloc[0])
                else:
                    df[col] = df[col].fillna(0)
                    
        # 6. Feature Engineering
        # Create BP_Category
        # normal < 120, prehypertension 120-139, stage 1 hypertension 140-159, stage 2 hypertension >= 160
        def get_bp_category(bp):
            if bp < 120: return 'Normal'
            elif bp < 140: return 'Prehypertension'
            elif bp < 160: return 'Stage 1 Hypertension'
            return 'Stage 2 Hypertension'
            
        df['BP_Category'] = df['trestbps'].apply(get_bp_category)
        
        # 7. Final Sanity Checks & Types
        df['age'] = df['age'].astype(int)
        df['trestbps'] = df['trestbps'].astype(int)
        df['chol'] = df['chol'].astype(int)
        df['thalach'] = df['thalach'].astype(int)
        df['ca'] = df['ca'].astype(int)
        df['thal'] = df['thal'].astype(int)
        df['target'] = df['target'].astype(int)
        
        # Record final shape
        stats['clean_rows'] = len(df)
        stats['clean_cols'] = len(df.columns)
        stats['clean_nulls'] = int(df.isna().sum().sum())
        
        # Save output
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Data cleaned and saved successfully to {self.output_path}!")
        return stats

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_in = os.path.join(script_dir, '..', 'data', 'raw_heart_disease.csv')
    clean_out = os.path.join(script_dir, '..', 'data', 'cleaned_heart_disease.csv')
    
    cleaner = DataCleaner(raw_in, clean_out)
    stats = cleaner.clean()
    print("Cleaning stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
