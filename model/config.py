from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
LABELS_PATH = ARTIFACTS_DIR / "label_list.json"
