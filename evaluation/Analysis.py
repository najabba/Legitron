import json
import glob
import os
import pandas as pd
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

THEME_KEYWORDS = {
    "Wounded, Sick & Dead": ["dead", "body", "bodies", "remains", "wounded", "sick", "medical", "hospital", "ambulance"],
    "Prisoners of War (POW)": ["prisoner", "captured", "detainee", "detention", "pow", "internment"],
    "Civilians": ["civilian", "protected person", "women", "children", "journalist", "humanitarian relief"],
    "Conduct of Hostilities": ["attack", "target", "weapon", "proportionality", "distinction", "military objective", "precautions"],
    "Occupation": ["occupation", "occupied", "occupying power"],
    "Emblems & Signs": ["emblem", "red cross", "red crescent", "flag", "insignia"],
}

def categorize_by_clustering(df, model_name="BAAI/bge-large-en", n_clusters=8):
    print("Embedding questions for clustering...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["question"].tolist(), show_progress_bar=True)
    
    print("Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster_id"] = kmeans.fit_predict(embeddings)
    
    # Optional: Name the clusters by finding the most central question
    cluster_names = {}
    for i in range(n_clusters):
        # Find question closest to center
        # (Simplified: just taking the first one for now, 
        # normally you'd calculate distance to centroid)
        sample_q = df[df["cluster_id"] == i]["question"].iloc[0]
        cluster_names[i] = f"Cluster {i}: {sample_q[:50]}..."
        
    df["theme"] = df["cluster_id"].map(cluster_names)
    return df

def load_data(files_pattern):
    data = []
    files = glob.glob(files_pattern)
    print(f"Found {len(files)} files to analyze.")
    
    for file_path in files:
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    items = content
                else:
                    items = [content]
                
                for item in items:
                    data.append({
                        "question": item.get("question", ""),
                        "correct": item.get("correct", False),
                        "file": os.path.basename(file_path)
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return pd.DataFrame(data)

def analyze_themes(df):
    # 1. Apply Categorization
    #df["theme"] = df["question"].apply(get_theme_by_keyword)
    
    # 2. Aggregate Data
    # Calculate Count, Accuracy (Mean of 'correct'), and Error Rate
    summary = df.groupby("theme").agg(
        count=("correct", "size"),
        accuracy=("correct", "mean")
    ).reset_index()
    
    # Add Error Rate (1 - Accuracy)
    summary["error_rate"] = 1.0 - summary["accuracy"]
    
    # Sort by Error Rate (Highest failures first)
    summary = summary.sort_values("error_rate", ascending=False)
    
    return summary


def main():
    # Load
    df = load_data("predictions/predictions/*.json")
    if df.empty:
        print("No data found!")
    
    print(f"Loaded {len(df)} questions.\n")
    
    # Analyze
    df = categorize_by_clustering(df)
    summary = analyze_themes(df)
    
    # Print Report
    print("=== Error Analysis by Theme ===")
    print(summary.to_string(formatters={
        'accuracy': '{:,.2%}'.format,
        'error_rate': '{:,.2%}'.format
    }, index=False))
    
    # Save
    summary.to_csv("error_analysis.csv", index=False)
    print(f"\nSaved detailed report to error_analysis.csv")

if __name__ == "__main__":
    main()