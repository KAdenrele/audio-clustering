import os
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Import your custom classes (assuming they are in these files)
from analyse_degradation import ForensicAudioEngine
from audio_processing import SocialAudioPipeline

# --- Setup ---
BASE_DIR = "./data/audio_files"
engine = ForensicAudioEngine()
results = []

# Define all the simulations you want to run
# Format: (Method Name, Friendly Name, Output Extension, Extra Args)
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

print("Starting Forensic Audit across 13 Social Media Pipelines...")

for model_name in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_path): continue
    
    print(f"\n--- Processing Model: {model_name} ---")
    
    # Process first 15 files per model to save time (adjust as needed)
    for filename in os.listdir(model_path)[:15]:
        if not filename.endswith(('.wav', '.mp3')): continue
        
        file_path = os.path.join(model_path, filename)
        
        try:
            # Initialize Pipeline for this specific file
            pipeline = SocialAudioPipeline(file_path)
            
            # Loop through ALL defined simulations
            for method_name, platform_name, ext, kwargs in SIMULATIONS:
                temp_output = f"temp_{filename}{ext}"
                
                try:
                    # 1. Run the specific social media simulation
                    # dynamically call the method: pipeline.process_tiktok(...)
                    method = getattr(pipeline, method_name)
                    method(temp_output, **kwargs)
                    
                    # 2. Run Forensic Analysis 
                    # Reference is the INPUT file (file_path) vs DEGRADED (temp_output)
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
                    # Always cleanup the temp file
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

X = df[features].fillna(0)

X_scaled = StandardScaler().fit_transform(X)

#using PCA
pca = PCA(n_components=2, random_state=42)
embedding = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_
total_var = sum(var_ratio) * 100

# # UMAP Projection
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
# embedding = reducer.fit_transform(X_scaled)

df['x'] = embedding[:, 0]
df['y'] = embedding[:, 1]

#Model Fingerprint
#Can we identify the Deepfake Model regardless of the platform?
fig_model = px.scatter(
    df, x='x', y='y', 
    color='model', 
    #symbol='category', # Circle for Voice, Diamond for Media
    title="Forensic Fingerprint: Model Stability Across All Platforms",
    hover_data=["filename", "platform", "lsd"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_model.write_html("data/cluster_by_model.html")


#Platform Degradation
#Which platforms destroy audio in similar ways?
fig_platform = px.scatter(
    df, x='x', y='y', 
    color='platform',
    symbol='category',
    title="Platform degradation Topology: Grouping by Compression Artifacts",
    hover_data=["filename", "model", "bandwidth_retained_pct"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig_platform.write_html("data/cluster_by_platform.html")
