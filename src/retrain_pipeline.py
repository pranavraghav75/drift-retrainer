import os
import subprocess

FLAG_PATH = "trigger_retrain.flag"

if os.path.exists(FLAG_PATH):
    print("Drift flag detected - retraining model...")

    result = subprocess.run(["python", "src/train.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Training script failed:")
        print(result.stderr)
        exit(1)

    os.remove(FLAG_PATH)
    print("Retraining complete — flag removed.")
else:
    print("No drift flag found. Skipping retrain.")
