"""Minimalny test połączenia z Weights & Biases (instrukcja — krok 4)."""

import os

import wandb
from dotenv import load_dotenv

load_dotenv()

print(f"Entity:  {os.getenv('WANDB_ENTITY')}")
print(f"Project: {os.getenv('WANDB_PROJECT')}")
print(f"Klucz:   {'skonfigurowany' if os.getenv('WANDB_API_KEY') else 'BRAK!'}")

wandb.init(
    project=os.getenv("WANDB_PROJECT", "test-asi"),
    entity=os.getenv("WANDB_ENTITY"),
)
wandb.log({"test_metric": 42})
wandb.finish()

print("Sprawdź dashboard na wandb.ai!")
