"""
Reusable functions to connect to Hopsworks and get the Feature Store
or Model Registry handle. Reads the API key from .env (never hardcode it here).
"""

import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


def get_project(project_name):
    """Logs in to Hopsworks and returns the project handle."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY not found. Check your .env file.")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=project_name,
    )
    return project


def get_feature_store(project_name):
    """Returns the Feature Store handle for the given project name."""
    project = get_project(project_name)
    return project.get_feature_store()


def get_model_registry(project_name):
    """Returns the Model Registry handle for the given project name."""
    project = get_project(project_name)
    return project.get_model_registry()