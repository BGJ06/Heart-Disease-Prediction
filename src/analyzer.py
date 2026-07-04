import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(data_path, plots_dir):
    """Generates static exploratory visualizations from the cleaned dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}")
        
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Configure matplotlib/seaborn styles for high-quality static charts
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    # Color palette
    colors = ["#0EA5E9", "#EF4444"] # Blue (No disease), Red (Heart disease)
    
    # 1. Scatter Plot: Age vs. Max Heart Rate (thalach) colored by target
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    scatter = sns.scatterplot(
        data=df, 
        x='age', 
        y='thalach', 
        hue='target', 
        palette=colors,
        alpha=0.8,
        s=70,
        edgecolor='w',
        linewidth=0.5,
        ax=ax1
    )
    ax1.set_title('Age vs. Maximum Heart Rate achieved by Disease Status')
    ax1.set_xlabel('Age (Years)')
    ax1.set_ylabel('Max Heart Rate achieved (bpm)')
    
    # Customize legend labels
    handles, labels = scatter.get_legend_handles_labels()
    ax1.legend(handles, ['Normal', 'Heart Disease'], title='Diagnosis', loc='upper right')
    
    plt.tight_layout()
    plot_path1 = os.path.join(plots_dir, 'age_vs_max_heart_rate.png')
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {plot_path1}")
    
    # 2. Correlation Matrix Heatmap
    # Filter numeric columns for correlation analysis
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target', 'ca', 'thal']
    corr_matrix = df[numeric_cols].corr()
    
    fig2, ax2 = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5, 
        square=True, 
        cbar_kws={"shrink": 0.8},
        ax=ax2
    )
    ax2.set_title('Correlation Matrix of Clinical Parameters')
    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, 'correlation_matrix.png')
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {plot_path2}")
    
    # 3. Clinical Distributions (Resting BP & Cholesterol)
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Resting BP Distribution
    sns.histplot(
        data=df, 
        x='trestbps', 
        hue='target', 
        multiple='stack', 
        palette=colors,
        kde=True,
        ax=ax3_1,
        bins=15
    )
    ax3_1.set_title('Resting Blood Pressure Distribution')
    ax3_1.set_xlabel('Resting Blood Pressure (mm Hg)')
    ax3_1.set_ylabel('Patient Count')
    ax3_1.legend(['Heart Disease', 'Normal'], title='Diagnosis')
    
    # Cholesterol Distribution
    sns.histplot(
        data=df, 
        x='chol', 
        hue='target', 
        multiple='stack', 
        palette=colors,
        kde=True,
        ax=ax3_2,
        bins=15
    )
    ax3_2.set_title('Serum Cholesterol Distribution')
    ax3_2.set_xlabel('Serum Cholesterol (mg/dl)')
    ax3_2.set_ylabel('Patient Count')
    ax3_2.legend(['Heart Disease', 'Normal'], title='Diagnosis')
    
    plt.tight_layout()
    plot_path3 = os.path.join(plots_dir, 'clinical_distributions.png')
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"Saved distribution plots to {plot_path3}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_data = os.path.join(script_dir, '..', 'data', 'cleaned_heart_disease.csv')
    plots_path = os.path.join(script_dir, '..', 'plots')
    generate_plots(clean_data, plots_path)
