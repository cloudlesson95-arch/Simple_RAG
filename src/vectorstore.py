import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#for local embedding
from langchain_huggingface import HuggingFaceEmbeddings #pip install sentence-transformers langchain-huggingface #~80MB model + torch
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, EMBEDDING_LOCAL_MODEL

from src.logging_config import setup_logging
logger = setup_logging(__name__)

def load_and_chunk_documents() -> list:
    """Load and chunk documents from the data directory

    Returns:
        list: A list of chunked documents ready for embedding.
    """
    files = ["fictional_text.txt", "cat-facts.txt", "pydantic.llms-full.txt"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.create_documents([text], metadatas=[{"source": filename}])    
        all_chunks.extend(chunks)

        logger.info(f"Loaded {filename}: Split into {len(chunks)}")

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def create_or_get_vectorstore(chunks=None, embeddings=None) -> Chroma:
    """Get or create the vector store.
    
    Args:
        chunks: List of documents to embed and store
        embeddings: Optional custom embeddings (uses default if None)
        
    Returns:
        Chroma: The vector store instance
    """
    if embeddings is None:
        # #gemini model through API
        # from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # embeddings = GoogleGenerativeAIEmbeddings(
        #     model = EMBEDDING_MODEL,
        #     google_api_key=os.getenv("GOOGLE_API_KEY")
        # )

        #local embedding model
        #make sure to remove 'chroma_db' folder on switch to avoid different embedding error 
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_LOCAL_MODEL)
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        logger.info(f"Loading existing database from {CHROMA_PERSIST_DIR}...")
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        if chunks != None:
            logger.info(f"Creating new database at {CHROMA_PERSIST_DIR}. This might take a while...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
        else:
            logger.warning(f"Database not found at {CHROMA_PERSIST_DIR}. And chunks are None to create one.")
            raise FileNotFoundError(f"No database found. Run 'python -m src.app index' first.")
    
    return vectorstore