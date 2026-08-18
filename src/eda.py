import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_dir="data", output_dir="plots", results_dir="results"):
    """
    Runs Exploratory Data Analysis (EDA) on the training files,
    saves resulting visualizations to output_dir, and logs statistics
    to a file in results_dir.
    """
    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    log_lines = []
    
    def log_print(msg=""):
        print(msg)
        log_lines.append(str(msg))
        
    log_print(f"=== Starting EDA script. Saving plots to '{output_dir}/' and text report to '{results_dir}/' ===\n")
    
    # 1. Load data
    train_path = os.path.join(data_dir, "train.csv")
    series_path = os.path.join(data_dir, "train_series.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(series_path):
        log_print(f"Error: Missing train files in '{data_dir}'. Make sure train.csv and train_series.csv are present.")
        return
        
    train_df = pd.read_csv(train_path)
    series_df = pd.read_csv(series_path)
    
    log_print(f"Loaded train.csv: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    log_print(f"Loaded train_series.csv: {series_df.shape[0]} rows, {series_df.shape[1]} columns\n")
    
    # 2. Target Labels Sparsity Analysis
    target_cols = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", 
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
    ]
    
    labeled_mask = train_df[target_cols].notnull().any(axis=1)
    num_labeled = labeled_mask.sum()
    num_unlabeled = len(train_df) - num_labeled
    
    log_print("--- Annotation Sparsity ---")
    log_print(f"Labeled studies (Clinician Ground Truth): {num_labeled} ({num_labeled/len(train_df)*100:.2f}%)")
    log_print(f"Unlabeled studies (Radiology Report Only): {num_unlabeled} ({num_unlabeled/len(train_df)*100:.2f}%)\n")
    
    # Plot Sparsity
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=["Labeled (Ground Truth)", "Unlabeled (Report Only)"], y=[num_labeled, num_unlabeled], palette="viridis")
    plt.title("Annotation Sparsity in train.csv")
    plt.ylabel("Count of Studies")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_sparsity.png"), dpi=150)
    plt.close()
    
    # 3. Label prevalence in labeled studies
    labeled_df = train_df[labeled_mask].copy()
    if num_labeled > 0:
        log_print("--- Target Class Distribution in Labeled Subset ---")
        for col in target_cols:
            pos_count = (labeled_df[col] == 1.0).sum()
            neg_count = (labeled_df[col] == 0.0).sum()
            log_print(f"  {col:<18} | Positive: {pos_count:>2} | Negative: {neg_count:>2}")
        log_print("")
        
        # Plot class prevalence
        melted_df = pd.melt(labeled_df[target_cols])
        melted_df["value"] = melted_df["value"].astype(str)
        plt.figure(figsize=(14, 6))
        sns.countplot(data=melted_df, x="variable", hue="value", palette="coolwarm")
        plt.xticks(rotation=45)
        plt.title(f"Distribution of Abnormalities in Labeled Studies (N = {num_labeled})")
        plt.xlabel("Abnormality Class")
        plt.ylabel("Count")
        plt.legend(title="Label", labels=["Negative (0)", "Positive (1)"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "label_prevalence.png"), dpi=150)
        plt.close()
        
    # 4. Text Report length analysis
    train_df["Report_Length"] = train_df["Report"].fillna("").apply(len)
    train_df["Word_Count"] = train_df["Report"].fillna("").apply(lambda x: len(x.split()))
    
    log_print("--- Report Statistics ---")
    log_print(f"Average character length: {train_df['Report_Length'].mean():.1f} (Median: {train_df['Report_Length'].median():.1f})")
    log_print(f"Average word count:      {train_df['Word_Count'].mean():.1f} (Median: {train_df['Word_Count'].median():.1f})\n")
    
    # Plot Report Lengths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(train_df["Report_Length"], bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Report Lengths (Characters)")
    axes[0].set_xlabel("Length")
    
    sns.histplot(train_df["Word_Count"], bins=50, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Report Word Counts")
    axes[1].set_xlabel("Words")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "report_lengths.png"), dpi=150)
    plt.close()
    
    # 5. Series Metadata analysis
    series_per_study = series_df.groupby("StudyInstanceUID").size()
    log_print("--- Series Metadata ---")
    log_print(f"Average series per study: {series_per_study.mean():.2f} (Min: {series_per_study.min()}, Max: {series_per_study.max()})")
    
    plane_counts = series_df["Anatomical_Plane"].value_counts()
    log_print("Anatomical Planes frequency:")
    for plane, count in plane_counts.items():
        log_print(f"  {plane:<10}: {count:>5} sequences")
    log_print("")
    
    # Plot series count distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(x=series_per_study, palette="muted")
    plt.title("Number of MRI Series per Study")
    plt.xlabel("Number of Series")
    plt.ylabel("Count of Studies")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "series_per_study.png"), dpi=150)
    plt.close()
    
    # Plot Plane & Contrast Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(data=series_df, x="Anatomical_Plane", ax=axes[0], palette="Set2")
    axes[0].set_title("Distribution of Planes")
    
    sns.countplot(data=series_df, x="Fluid_Sensitive", ax=axes[1], palette="pastel")
    axes[1].set_title("Fluid Sensitive Sequences")
    
    sns.countplot(data=series_df, x="Fat_Suppression", ax=axes[2], palette="pastel")
    axes[2].set_title("Fat Suppression Sequences")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plane_and_contrast_distribution.png"), dpi=150)
    plt.close()
    
    # Fluid vs Fat Suppression cross tab
    cross_tab = pd.crosstab(series_df["Fluid_Sensitive"], series_df["Fat_Suppression"])
    log_print("Correlation between Fluid Sensitive and Fat Suppression:")
    log_print(str(cross_tab))
    
    # Save text log to the results directory
    log_file_path = os.path.join(results_dir, "eda_results.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        
    log_print(f"\n=== EDA complete. Log saved to '{log_file_path}' ===")

if __name__ == "__main__":
    run_eda()
