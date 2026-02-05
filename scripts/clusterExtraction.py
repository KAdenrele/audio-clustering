import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px

# Setup
BASE_DIR = "./data/audio_files"
features_list = []

def extract_blind_features(file_path):
    """Extracts forensic features without needing a reference file."""
    y, sr = librosa.load(file_path, sr=None) # Load at native rate
    
    # 1. Spectral Features
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # 2. MFCCs (The 'Fingerprint')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    
    # Combine into a single dictionary
    feat_dict = {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "flatness": flatness,
        "rolloff": rolloff,
        "zcr": zcr
    }
    # Add MFCCs as separate columns
    for i, m in enumerate(mfcc_means):
        feat_dict[f"mfcc_{i}"] = m
        
    return feat_dict

# Scan directories
for model_name in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_path): continue
    
    print(f"Processing Model: {model_name}...")
    for file in os.listdir(model_path):
        if file.endswith(('.wav', '.mp3', '.mp4')):
            try:
                f_path = os.path.join(model_path, file)
                feats = extract_blind_features(f_path)
                feats["model"] = model_name
                feats["filename"] = file
                features_list.append(feats)
            except Exception as e:
                print(f"Error on {file}: {e}")

df = pd.DataFrame(features_list)

# --- Normalisation & Clustering ---
# Drop non-numeric columns for calculation
X = df.drop(columns=["model", "filename"])
X_scaled = StandardScaler().fit_transform(X)

# t-SNE reduction
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
projections = tsne.fit_transform(X_scaled)
df['x'], df['y'] = projections[:, 0], projections[:, 1]

fig = px.scatter(
    df, x='x', y='y', 
    color='model',
    hover_data=['filename', 'centroid', 'flatness'],
    title="Forensic Cluster Analysis (No-Reference)",
    labels={'x': 'Feature Dimension 1', 'y': 'Feature Dimension 2'},
    template="plotly_dark"
)

# Enhance visualization
fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(dragmode='lasso') # Allow for manual selection of clusters

fig.write_html("audio_clusters.html")
print("Analysis complete! Open audio_clusters.html in your browser.")