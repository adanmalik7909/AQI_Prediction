"""
verify_environment.py
------------------------
Quick sanity check: imports every critical library used by the
feature/training pipeline, and reports versions. Run this FIRST after
any environment setup/change, before running the actual pipeline -
catches dependency conflicts in seconds instead of minutes.

Run: python verify_environment.py
"""

import sys

print(f"Python version: {sys.version}\n")

checks = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("tensorflow", "tensorflow"),
    ("shap", "shap"),
    ("hopsworks", "hopsworks"),
    ("google.protobuf", "protobuf"),
    ("joblib", "joblib"),
    ("dotenv", "python-dotenv"),
]

all_ok = True

for module_name, display_name in checks:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK]   {display_name:<20} {version}")
    except Exception as e:
        all_ok = False
        print(f"[FAIL] {display_name:<20} {type(e).__name__}: {e}")

print()
if all_ok:
    print("All critical libraries imported successfully. Environment looks good!")
else:
    print("Some libraries failed to import - fix these BEFORE running the pipeline.")