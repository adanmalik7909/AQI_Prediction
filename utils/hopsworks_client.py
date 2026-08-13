"""
Reusable functions to connect to Hopsworks and get the Feature Store
or Model Registry handle. Reads the API key from .env (never hardcode it here).

Includes retry logic because the EU-West Hopsworks server can occasionally
drop connections (especially from long-distance clients).
"""

import os
import time
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

MAX_RETRIES = 3
RETRY_DELAYS = [5, 10, 20]  # seconds between retries (exponential backoff)


def get_project(project_name):
    """Logs in to Hopsworks and returns the project handle.
    Retries up to MAX_RETRIES times on connection failures."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY not found. Check your .env file.")

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=project_name,
            )
            return project
        except (ConnectionError, TimeoutError, OSError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                print(f"  Hopsworks connection failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                      f"{type(e).__name__}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  All {MAX_RETRIES + 1} connection attempts failed.")
                raise last_error


def get_feature_store(project_name):
    """Returns the Feature Store handle for the given project name."""
    project = get_project(project_name)
    return project.get_feature_store()


def get_model_registry(project_name):
    """Returns the Model Registry handle for the given project name."""
    project = get_project(project_name)
    return project.get_model_registry()