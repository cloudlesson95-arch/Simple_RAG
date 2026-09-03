import os
from dotenv import load_dotenv
load_dotenv()

#Text chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#Retrieval parameters
K_RETRIEVAL = 4
K_EVALUATION = 10

#Agent parameters
MAX_RETRIES = 2

#Model used
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_LOCAL_MODEL = "all-MiniLM-L6-v2"
MAIN_LLM_MODEL = "groq" #or gemini
EVAL_LLM_MODEL = "groq" #or gemini

#Database paths
CHROMA_PERSIST_DIR = "./chroma_db"
DATA_DIR = "data"

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_CONSOLE = True 
LOG_TO_FILE = True 
LOG_FILE_PATH = "app.log" 
LOG_FORMAT = '%(message)s'#'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
CAPTURE_EXTERNAL_LOGS = False  # Capture logs from external libraries

ROUTING_METHOD = os.getenv("ROUTING_METHOD", "classical")  # "llm", "classical"
GENERATE_VISUALIZATION = True
CLUSTERS_DIR = "clusters"
CLASSIFIER_MODEL_PATH = f"{CLUSTERS_DIR}/retrieval_classifier.joblib"

ENABLE_SEMANTIC_CACHE = True
CACHE_SIMILARITY_THRESHOLD = 0.95
SEMANTIC_CACHE_DIR = "s_cache"

# Preferred models per provider (tried in order, first available wins)
PREFERRED_MODELS = {
    "groq": ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
    "gemini": ["gemini-2.5-flash", "gemini-3.5-flash"],
}

# Baseline configuration
EVAL_QUESTIONS_PATH = "baseline/questions.json"

DB_PATH = "rag.db"
