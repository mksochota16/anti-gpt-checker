import os

from pathlib import Path
from typing import List

import yaml
from pydantic import ValidationError

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from api.analysis_task_queue import AnalysisTaskQueue
from api.api_models.lightbulb_score import LightbulbScoreConfig

load_dotenv()


API_MONGODB_URI = os.getenv("API_MONGODB_URI")
API_MONGODB_PORT = int(os.getenv("API_MONGODB_PORT"))
API_MONGODB_DB_NAME= os.getenv("API_MONGODB_DB_NAME")

API_MONGODB_AUTH_USER = os.getenv("API_MONGODB_AUTH_USER")
API_MONGODB_AUTH_PASS = os.getenv("API_MONGODB_AUTH_PASS")
API_MONGODB_AUTH_DB = os.getenv("API_MONGODB_AUTH_DB")

API_ADMIN_USER_ID = os.getenv("API_ADMIN_USER_ID", "-1")

if API_MONGODB_AUTH_USER and API_MONGODB_AUTH_PASS and API_MONGODB_AUTH_DB:
    API_MONGO_CLIENT = MongoClient(host=API_MONGODB_URI, port=API_MONGODB_PORT, username=API_MONGODB_AUTH_USER, password=API_MONGODB_AUTH_PASS, authSource=API_MONGODB_AUTH_DB)
    API_ASYNC_MONGO_CLIENT = AsyncIOMotorClient(host=API_MONGODB_URI, port=API_MONGODB_PORT, username=API_MONGODB_AUTH_USER, password=API_MONGODB_AUTH_PASS, authSource=API_MONGODB_AUTH_DB)
else:
    API_MONGO_CLIENT = MongoClient(API_MONGODB_URI, API_MONGODB_PORT)
    API_ASYNC_MONGO_CLIENT = AsyncIOMotorClient(API_MONGODB_URI, API_MONGODB_PORT)

API_ANALYSIS_COLLECTION_NAME = os.getenv("API_ANALYSIS_COLLECTION_NAME", "analysis")
API_DOCUMENTS_COLLECTION_NAME = os.getenv("API_DOCUMENTS_COLLECTION_NAME", "documents")
API_USERS_COLLECTION_NAME = os.getenv("API_USERS_COLLECTION_NAME", "users")
API_ATTRIBUTES_COLLECTION_NAME = os.getenv("API_ATTRIBUTES_COLLECTION_NAME", "attributes")
API_ATTRIBUTES_REFERENCE_COLLECTION_NAME = os.getenv("API_ATTRIBUTES_REFERENCE_COLLECTION_NAME", "attributes_reference")
API_LIGHTBULBS_SCORES_COLLECTION_NAME = os.getenv("API_ATTRIBUTES_REFERENCE_COLLECTION_NAME", "lightbulb_scores")

API_WEB_APP_IP = os.getenv("API_WEB_APP_IP", "")
API_WEB_APP_PORT = int(os.getenv("API_WEB_APP_PORT", 8000))

API_HISTOGRAMS_PATH = os.getenv("API_HISTOGRAMS_PATH")

API_LIGHTBULBS_SCORES_CONFIG_PATH = os.getenv("API_LIGHTBULBS_SCORES_CONFIG_PATH")
API_MOST_IMPORTANT_ATTRIBUTES_CONFIG_PATH = os.getenv("API_MOST_IMPORTANT_ATTRIBUTES_CONFIG_PATH")
API_FAKE_SCORE_FEATURES_CONFIG_PATH = os.getenv("API_FAKE_SCORE_FEATURES_CONFIG_PATH")

API_CATBOOST_MODEL_PATH = os.getenv("API_CATBOOST_MODEL_PATH")
API_CATBOOST_VECTORIZER = os.getenv("API_CATBOOST_VECTORIZER")

LLM_REFERENCE_MODEL = os.getenv("LLM_REFERENCE_MODEL", None)

def load_lightbulbs_scores_parameters() -> dict[str, LightbulbScoreConfig]:
    path = Path(API_LIGHTBULBS_SCORES_CONFIG_PATH)
    with path.open("r", encoding="utf‑8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Top level YAML element must be a list")

    try:
        return {LightbulbScoreConfig(**item).attribute_name: LightbulbScoreConfig(**item) for item in data}
    except ValidationError as exc:
        # Pretty print validation errors then re‑raise
        print("Config file validation failed:\n", exc)
        raise

def load_attribute_names_list(filepath: str) -> List[str]:
    path = Path(filepath)
    if path.suffix.lower() in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or []
        if not isinstance(data, list):
            raise ValueError("YAML file must contain a list at the top level")
        return [str(item) for item in data]
    else:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]

API_LIGHTBULBS_SCORES_PARAMETERS: dict[str, LightbulbScoreConfig] = load_lightbulbs_scores_parameters()
API_MOST_IMPORTANT_ATTRIBUTES: List[str] = load_attribute_names_list(API_MOST_IMPORTANT_ATTRIBUTES_CONFIG_PATH)
API_FAKE_SCORE_FEATURES: List[str] = load_attribute_names_list(API_FAKE_SCORE_FEATURES_CONFIG_PATH)

API_SHARED_SECRET_KEY = os.getenv("API_SHARED_SECRET_KEY")

API_DEBUG = (os.getenv("API_DEBUG", "False").lower() in ["1", "true", "t", "yes", "y"])
API_DEBUG_USER_ID = os.getenv("API_DEBUG_USER_ID", "0000000000000000000000000")

API_MAX_CONCURRENT_TASKS = int(os.getenv("API_MAX_CONCURRENT_TASKS", "1"))
ANALYSIS_TASK_QUEUE: AnalysisTaskQueue | None = None
def init_analysis_task_queue() -> None:
    global ANALYSIS_TASK_QUEUE
    ANALYSIS_TASK_QUEUE = AnalysisTaskQueue(API_MAX_CONCURRENT_TASKS)