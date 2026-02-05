import os
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Import your custom modules
from analyse_degradation import ForensicAudioEngine
from audio_processing import SocialAudioPipeline
from audio_reverse import AudioRestorer

# --- Configuration ---
BASE_DIR = "./data/audio_files"
RESULTS_DIR = "./data"
TARGET_PLATFORM = "whatsapp_voice"
engine = ForensicAudioEngine()
results = []

print(f"🚀 Starting Signature Persistence Test...")
print(f"   Attack: {TARGET_PLATFORM}")
print(f"   Recovery: AudioRestorer.restore_for_metrics()")

for model_name in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_path): continue
    
    print(f"\n--- Auditing Model Identity: {model_name} ---")
    
    # Process 15 files per model
    for filename in os.listdir(model_path)[:15]:
        if not filename.endswith(('.wav', '.mp3')): continue
        
        orig_path = os.path.join(model_path, filename)
        
        # Temp filenames
        degraded_path = f"temp_attack_{filename}.ogg"
        restored_path = f"temp_reversed_{filename}.wav"
        
        try:
            # 1. ATTACK (Degrade)
            pipeline = SocialAudioPipeline(orig_path)
            getattr(pipeline, f"process_{TARGET_PLATFORM}")(degraded_path)
            
            # 2. REVERSE (Restore)
            # We use the degraded file as input
            restorer = AudioRestorer(degraded_path, platform=TARGET_PLATFORM)
            restorer.restore_for_metrics() # Use your optimized restoration
            restorer.save(restored_path)
            
            # 3. COMPARE: Unprocessed Original vs. Reversed Audio
            # We are measuring "What signature survives?"
            report = engine.run_full_analysis(orig_path, restored_path)
            
            if report["status"] == "success":
                metrics = report["metrics"]
                metrics["model"] = model_name
                metrics["filename"] = filename
                # Capture the alignment score to ensure we aren't just clustering misaligned files
                metrics["align_conf"] = metrics.get("alignment_confidence", 0)
                results.append(metrics)

        except Exception as e:
            print(f"   Skipped {filename}: {e}")
            
        finally:
            # Cleanup
            if os.path.exists(degraded_path): os.remove(degraded_path)
            if os.path.exists(restored_path): os.remove(restored_path)

# --- Clustering Logic ---
print("\n Calculating Forensic Fingerprints...")
df = pd.DataFrame(results)

# Filter out bad alignments (critical for this specific test)
# If alignment is bad, the metrics are random noise, not model signatures.
df_clean = df[df["align_conf"] > 0.8].copy()
print(f"   Retained {len(df_clean)} samples with valid alignment.")

# Features used to define the "Signature"
features = [
    "lsd", 
    "si_sdr", 
    "centroid_shift_hz", 
    "bandwidth_retained_pct", 
    "flatness_change",
    "avg_coherence"
]

# Impute & Scale
X = df_clean[features].fillna(df_clean[features].mean())
X_scaled = StandardScaler().fit_transform(X)

# UMAP Projection
# n_neighbors=7 is tuned for ~15 samples per class to allow distinct islands
reducer = umap.UMAP(n_neighbors=7, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

df_clean['x'] = embedding[:, 0]
df_clean['y'] = embedding[:, 1]

# --- Visualization ---
fig = px.scatter(
    df_clean, x='x', y='y', 
    color='model', # Color by Source Model
    title=f"Signature Persistence: Model Identity After {TARGET_PLATFORM} Reversal",
    hover_data=["filename", "lsd", "flatness_change"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Bold,
    labels={'x': 'Forensic Dimension 1', 'y': 'Forensic Dimension 2'}
)

fig.add_annotation(
    text="Tight clusters = High forensic persistence.<br>Overlapping clusters = Signature lost.",
    xref="paper", yref="paper",
    x=0, y=1.05, showarrow=False,
    font=dict(color="gray", size=12)
)

output_file = os.path.join(RESULTS_DIR, "data/signature_persistence_cluster.html")
fig.write_html(output_file)
print(f"Analysis Complete. Open {output_file} to view clusters.")