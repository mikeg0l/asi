import os
import sys

import wandb
from dotenv import load_dotenv

load_dotenv()

entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT":W)

artifact_path = f"{entity}/{project}/baseline-model:latest"

api = wandb.Api()
artifact = api.artifact(artifact_path)
model_dir = artifact.download()
print(f"Model pobrany do: {model_dir}")
