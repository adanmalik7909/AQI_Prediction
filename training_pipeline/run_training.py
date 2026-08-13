"""
training_pipeline/run_training.py
------------------------------------
Main entry point for the training pipeline. Runs both steps in order:
  1. train_models    -> train + compare 4 models x 3 horizons, save locally
  2. register_model  -> push the WINNING model per horizon to the
                         Hopsworks Model Registry

This is the ONE file that GitHub Actions will call once a day
(see Phase 9 - Automation).
"""

from train_models import main as train_main
from register_model import main as register_main


def main():
    print("=" * 65)
    print("STEP 1: Training models")
    print("=" * 65)
    train_main()

    print("\n" + "=" * 65)
    print("STEP 2: Registering winning models to Model Registry")
    print("=" * 65)
    register_main()


if __name__ == "__main__":
    main()