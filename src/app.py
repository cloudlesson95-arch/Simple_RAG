import argparse
import os
import shutil
from dotenv import load_dotenv
from src.logging_config import setup_logging
from src.config import MAIN_LLM_MODEL, CHROMA_PERSIST_DIR
 
load_dotenv()
logger = setup_logging(__name__)

def main():
    parser = argparse.ArgumentParser(description="RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Build/update document index")
    index_parser.add_argument("--rebuild", action="store_true", help="Force rebuild of index")
    
    # Query command  
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation tests")
    
    args = parser.parse_args() 

    if args.command == "index":
        from src.vectorstore import load_and_chunk_documents, create_or_get_vectorstore
        
        if args.rebuild:
            logger.info(f"Rebuilding index - removing existing database at path: {CHROMA_PERSIST_DIR}")
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)
                logger.info(f"Removed {CHROMA_PERSIST_DIR}")
        
        chunks = load_and_chunk_documents()
        vectorstore = create_or_get_vectorstore(chunks)
        logger.info("Index created/loaded successfully")
        
    elif args.command == "query":
        from src.vectorstore import create_or_get_vectorstore
        from src.rag_agent import setup_router, answer_question
        from src.utils import create_llm
        router = setup_router()
        vectorstore = create_or_get_vectorstore()
        answer_llm = create_llm(MAIN_LLM_MODEL)
        answer = answer_question(args.question, router, vectorstore, answer_llm)
        print(answer)
            
    elif args.command == "evaluate":
        from src.evaluator import run_evaluation
        run_evaluation()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()