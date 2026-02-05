import subprocess
import sys
import os

def run_task(script_path):
    """Runs a python script located in the scripts/ folder."""
    full_path = os.path.join("scripts", script_path)
    
    print(f"\n [MAIN] Starting task: {full_path}")
    
    if not os.path.exists(full_path):
        print(f" [MAIN] Error: File not found at {full_path}")
        sys.exit(1)

    try:
        # sys.executable ensures we use the same 'uv' python environment
        subprocess.run([sys.executable, full_path], check=True)
        print(f" [MAIN] Task complete: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f" [MAIN] Task failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    #run_task("audio_download.py")
    run_task("ProcessingCluster.py") 
    run_task("SignaturePersistence.py")
    
    print("\n🎉 [MAIN] Pipeline finished successfully. Outputs saved to data/ folder.")