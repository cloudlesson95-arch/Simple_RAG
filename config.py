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
MAIN_LLM_MODEL = "gemini" #or groq
AGENT_LLM_MODEL = "groq" #or gemini
EVAL_LLM_MODEL = "gemini" #or groq

#Database paths
CHROMA_PERSIST_DIR = "./chroma_db"
DATA_DIR = "data"
