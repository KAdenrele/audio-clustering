import os
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import your custom classes
from analyse_degradation import ForensicAudioEngine
from audio_processing import SocialAudioPipeline
# Use explicit string path to avoid import errors from scripts.signaturePersistence
RESULTS_DIR = "data" 

# --- Setup ---
BASE_DIR = "./data/audio_files"
engine = ForensicAudioEngine()
results = []

# Simulations Configuration
SIMULATIONS = [
    ("process_tiktok", "TikTok Voice", ".mp4", {}),
    ("process_instagram", "Instagram Voice", ".mp4", {}),
    ("process_whatsapp_voice", "WhatsApp Voice", ".ogg", {}),
    ("process_facebook", "Facebook Voice", ".mp4", {}),
    ("process_signal", "Signal Voice", ".ogg", {}),
    ("process_telegram", "Telegram Voice", ".ogg", {}),
    # ("process_whatsapp_media", "WhatsApp Media (Std)", ".mp4", {"quality_mode": "standard"}),
    # ("process_whatsapp_media", "WhatsApp Media (HD)", ".mp4", {"quality_mode": "high"}),
    # ("process_telegram_media", "Telegram Media", ".mp4", {}),
    # ("process_signal_media", "Signal Media", ".mp4", {}),
    # ("process_tiktok_media", "TikTok Media", ".mp4", {}),
    # ("process_instagram_media", "Instagram Media", ".mp4", {}),
    # ("process_facebook_media", "Facebook Media", ".mp4", {})
]

print("Starting Forensic Audit across Social Media Pipelines...")

for model_name in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_path): continue
    
    print(f"\n--- Processing Model: {model_name} ---")
    
    for filename in os.listdir(model_path)[:15]:
        if not filename.endswith(('.wav', '.mp3')): continue
        file_path = os.path.join(model_path, filename)
        
        try:
            pipeline = SocialAudioPipeline(file_path)
            
            for method_name, platform_name, ext, kwargs in SIMULATIONS:
                temp_output = f"temp_{filename}{ext}"
                try:
                    # 1. Run Simulation
                    getattr(pipeline, method_name)(temp_output, **kwargs)
                    
                    # 2. Run Forensic Analysis
                    report = engine.run_full_analysis(file_path, temp_output)
                    
                    if report["status"] == "success":
                        metrics = report["metrics"]
                        metrics["model"] = model_name
                        metrics["filename"] = filename
                        metrics["platform"] = platform_name
                        metrics["category"] = "Media" if "Media" in platform_name else "Voice"
                        results.append(metrics)
                    
                except Exception as e:
                    print(f"  Failed {platform_name} on {filename}: {e}")
                finally:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                        
        except Exception as e:
            print(f"Critical error loading {filename}: {e}")

# --- Clustering Logic ---
print("\nCalculating Clusters...")
df = pd.DataFrame(results)

features = [
    "lsd", 
    "si_sdr", 
    "centroid_shift_hz", 
    "bandwidth_retained_pct", 
    "flatness_change",
    "avg_coherence"
]

# FIX 1: Mean Imputation instead of 0
# 0 is a valid value for some metrics (LSD=Perfect), so filling NaNs with 0 destroys data integrity.
X = df[features].fillna(df[features].mean())

X_scaled = StandardScaler().fit_transform(X)

# FIX 2: Tuned Neighbors for Small Data
# n_neighbors=6 is better for ~15 samples/class. Prevents forced merging of distinct models.
reducer = umap.UMAP(n_neighbors=6, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

df['x'] = embedding[:, 0]
df['y'] = embedding[:, 1]

# 1. Model Fingerprint Plot
fig_model = px.scatter(
    df, x='x', y='y', 
    color='model', 
    title="Forensic Fingerprint: Model Stability Across All Platforms",
    hover_data=["filename", "platform", "lsd"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Bold
)
# Ensure directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
fig_model.write_html(os.path.join(RESULTS_DIR, "cluster_by_model.html"))

# 2. Platform Degradation Plot
fig_platform = px.scatter(
    df, x='x', y='y', 
    color='platform',
    symbol='category',
    title="Platform Degradation Topology",
    hover_data=["filename", "model", "bandwidth_retained_pct"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig_platform.write_html(os.path.join(RESULTS_DIR, "cluster_by_platform.html"))

# --- PCA Analysis ---
pca = PCA(n_components=2, random_state=42)
embedding_pca = pca.fit_transform(X_scaled)

var_ratio = pca.explained_variance_ratio_
total_var = sum(var_ratio) * 100
print(f"   PCA Variance Explained: {total_var:.2f}% (PC1: {var_ratio[0]:.2f}, PC2: {var_ratio[1]:.2f})")

# Overwrite coordinates for PCA Plot
df['x_pca'] = embedding_pca[:, 0]
df['y_pca'] = embedding_pca[:, 1]

fig_pca = px.scatter(
    df, x='x_pca', y='y_pca', 
    color='model', 
    title=f"PCA Analysis (Variance Explained: {total_var:.1f}%)",
    hover_data=["filename", "lsd", "flatness_change"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Bold,
    labels={'x_pca': f'PC1 ({var_ratio[0]*100:.1f}%)', 'y_pca': f'PC2 ({var_ratio[1]*100:.1f}%)'}
)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

for i, feature in enumerate(features):
    fig_pca.add_annotation(
        x=loadings[i, 0] * 3, 
        y=loadings[i, 1] * 3,
        text=feature,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363"
    )

fig_pca.write_html(os.path.join(RESULTS_DIR, "cluster_by_model_pca.html"))
print("✅ Analysis Complete.")