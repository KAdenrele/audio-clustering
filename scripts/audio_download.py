import os
import time
import io
import itertools
import soundfile as sf
from datasets import load_dataset, Features, Value, Sequence
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("hf_token")
DATA_ROOT = "/app/data"
classes = ["WF1", "WF2", "WF3", "WF4", "WF5", "WF6", "WF7", "R"]

# 1. Define the schema MANUALLY.
# By strictly defining 'audio' as a struct of bytes/string, we prevent
# the 'Audio' feature class from ever initializing.
strict_features = Features({
    "audio": {
        "bytes": Value("binary"),
        "path": Value("string")
    },
    "audio_id": Value("string"),
    "real_or_fake": Value("string")
})

print("Initializing dataset stream (Bypassing Audio Feature)...")

try:
    # 2. Load with specific features
    base_ds = load_dataset(
        "ajaykarthick/wavefake-audio",
        split="train",
        streaming=True,
        token=hf_token,
        features=strict_features  # <--- This is the magic fix
    )

    for class_name in classes:
        print(f"\n--- Processing class: {class_name} ---")
        output_dir = os.path.join(DATA_ROOT, "audio_files", class_name)
        os.makedirs(output_dir, exist_ok=True)

        filtered_ds = base_ds.filter(lambda x: x["real_or_fake"] == class_name)

        print(f"Downloading 20 samples...")
        
        for i, example in enumerate(itertools.islice(filtered_ds, 20)):
            try:
                # 3. Access raw bytes directly
                audio_bytes = example["audio"]["bytes"]
                filename = example['audio_id']
                
                if audio_bytes is None:
                    print(f"  Skipping {filename}: No audio bytes.")
                    continue

                # 4. Decode with SoundFile (Pure CPU, no Torch)
                with io.BytesIO(audio_bytes) as buffer:
                    data, samplerate = sf.read(buffer)
                
                output_path = os.path.join(output_dir, f"{filename}.wav")
                
                if not os.path.exists(output_path):
                    sf.write(output_path, data, samplerate)
                    print(f"  Saved {i+1}: {filename} (Rate: {samplerate}Hz)")
                else:
                    print(f"  Skipped {i+1}: {filename} (Exists)")
                
                time.sleep(0.1)

            except Exception as e:
                print(f"  Error processing sample {i+1}: {e}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")