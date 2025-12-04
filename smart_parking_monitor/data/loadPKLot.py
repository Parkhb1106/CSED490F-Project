# python3 loadPKLot.py

import os
from pathlib import Path
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = str(Path.cwd())

path = kagglehub.dataset_download("ammarnassanalhajali/pklot-dataset")
print("Path:", path)
