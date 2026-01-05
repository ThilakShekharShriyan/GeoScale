import yaml
from pathlib import Path

def load_config(config_path="configs/default.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)
