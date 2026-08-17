from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import K_EVALUATION, MAIN_LLM_MODEL, EVAL_LLM_MODEL, EMBEDDING_LOCAL_MODEL, CHROMA_PERSIST_DIR
from src.utils import create_llm
from src.vectorstore import create_or_get_vectorstore
from src.rag_agent import setup_router, answer_question

from src.logging_config import setup_logging
logger = setup_logging(__name__)

def load_questions(filepath: str) -> list[dict]:
    """Parse the question.txt file into a list of dictionaries.
    
    Args:
        filepath: Path to the questions file.
        
    Returns:
        list[dict]: List of question dictionaries with 'query', 'expected_context', 
                   and 'expected_answer' keys.
    """
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_q = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Query:"):
            # Handle the [adversarial] tag 
            q_text = line.replace("Query:", "").replace("[adversarial]", "").strip()
            if q_text.startswith('"') and q_text.endswith('"'):
                current_q["query"] = q_text[1:-1]
        elif line.startswith('"') and "query" not in current_q:
            # Handle multi-line query where text is on the next line
            current_q["query"] = line.strip('"')
        elif line.startswith("Expected Context:"):
            try:
                context = line.split('"')[1]
                current_q["expected_context"] = context
            except IndexError:
                current_q["expected_context"] = line.replace("Expected Context:", "").strip()
        elif line.startswith("Expected Answer:"):
            current_q["expected_answer"] = line.replace("Expected Answer:", "").strip()

            if "query" in current_q and "expected_context" in current_q:
                questions.append(current_q)
            current_q = {}
    return questions  


def run_evaluation():
    logger.info("Loading vector database")
    vectorstore = create_or_get_vectorstore()
    router = setup_router()
    answer_llm = create_llm(MAIN_LLM_MODEL)

    questions = load_questions("baseline/question.txt")
    logger.info(f"Loaded {len(questions)} test questions.\n")

    successful_retrievals = 0
    judge_llm = create_llm(EVAL_LLM_MODEL)
    
    for i,q in enumerate(questions):
        logger.info(f"[{i+1}/{len(questions)}] Testing: '{q['query']}'")

        try:
            answer = answer_question(q['query'], router, vectorstore, answer_llm)
            logger.info(f"\tThe user asked: {q['query']}")
            logger.info(f"\tLLM answer: {answer}")

            judge_prompt = f"""You are an impartial judge evaluating a Search engine.
            The user asked: '{q['query']}'
            The system retrieved this context:
            {answer}

            Does this answer correctly address the question?
            Reply ONLY with "YES" or "NO". Do not explain."""

            response = judge_llm.invoke(judge_prompt)
            decision = response.content.strip().upper()

            if "YES" in decision:
                logger.info("\tSUCCESS: LLM confirmed.")
                successful_retrievals +=1
            else:
                logger.info("\tFAIL: LLM denied")
                logger.info(f"\tExpected to find: {q['expected_answer'][:100]}...")
                logger.info("-"*30)
        except Exception as e:
            logger.error(f"\tERROR: {str(e)}")

    precision = (successful_retrievals/len(questions)) * 100
    logger.info(f"\nFinal score: Precision@4 - {precision:.1f}%")
    logger.info(f"({successful_retrievals} out of {len(questions)} retrieved correctly)")
