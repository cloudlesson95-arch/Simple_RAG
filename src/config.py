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

ROUTING_METHOD = os.getenv("ROUTING_METHOD", "llm")  # "llm", "classical"
GENERATE_VISUALIZATION = True
CLUSTERS_DIR = "clusters"
CLASSIFIER_MODEL_PATH = f"{CLUSTERS_DIR}/retrieval_classifier.joblib"
