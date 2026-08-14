import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from src.config import K_RETRIEVAL, MAX_RETRIES, EMBEDDING_LOCAL_MODEL, AGENT_LLM_MODEL, CHROMA_PERSIST_DIR
from src.utils import create_llm
from src.logging_config import setup_logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq #back-up llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logger = setup_logging(__name__)

class RouteDecision(BaseModel):
    """Decides which data source to use based on the user's query."""

    source: Literal["pydantic.llms-full.txt", "cat-facts.txt", "fictional_text.txt", "none"] = Field(
        description="""Which data source:
        - 'pydantic.llms-full.txt': questions about Python, AI, Pydantic, LLMs, agents, models
        - 'cat-facts.txt': questions about cats, animals, pets
        - 'fictional_text.txt': questions about Oakhaven, pumpkin festival, city council, Elena Rostova
        - 'none': math, greetings, or anything not in the above"""
    )

    reasoning: str = Field(
        description="A short 1-sentence explanation of why you chose this source."
    )

def setup_router():
    """Setup the router LLM for determining data sources.
    
    Returns:
        The router LLM configured with structured output for RouteDecision.
    """    
    llm = create_llm(AGENT_LLM_MODEL)

    router_llm = llm.with_structured_output(RouteDecision)
    return router_llm

def load_vectorstore() -> Chroma:
    """Load the existing vector database from disk.
    
    Returns:
        Chroma: The loaded vector store instance.
    """    
    logger.info("Loading vector database")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_LOCAL_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    return vectorstore


def answer_question(question: str, router, vectorstore, answer_llm, max_retries: int = MAX_RETRIES) -> str:
    """Answer a user question using RAG with self-correction.
    
    Args:
        question: The user's question to answer.
        router: The router LLM for determining data source.
        vectorstore: The vector store for retrieval.
        answer_llm: The LLM for generating answers.
        max_retries: Maximum number of rephrasing attempts (default: from config).
        
    Returns:
        str: The answer to the user's question.
    """
    decision = router.invoke(question)
    logger.info(f"[Router]: '{decision.source}' ({decision.reasoning})")

    if decision.source == "none":
        logger.info("[Router] No retrieval needed")
        response = answer_llm.invoke(question)
        return response.content

    def retrieve_and_answer(query):
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": K_RETRIEVAL,
                "filter": {"source": decision.source}
            }
        )
        results = retriever.invoke(query)
        context_text = "\n---\n".join([doc.page_content for doc in results])

        prompt = f"""Answer ONLY based on the provided context.
        If the answer is not in the context, say "I don't know."

        Context:
        {context_text}

        Question: {query}
        Answer:"""
        return answer_llm.invoke(prompt).content

    current_query = question
    answer = retrieve_and_answer(current_query)
    
    #Self-Correction loop
    for attempt in range(max_retries):
        if "I don't know" in answer:
            logger.info(f"[Agent] Attempt {attempt + 1}/{max_retries}: answer insufficient, rephrasing...")

            rephrase_prompt = f"""The user asked: "{current_query}"
            A search returned no useful results. 
            Rephrase this question using different keywords.
            Return ONLY the rephrased question."""

            current_query = answer_llm.invoke(rephrase_prompt).content.strip()
            logger.info(f"[Agent] Rephrased query: '{current_query}'")

            answer = retrieve_and_answer(current_query)

    return answer

if __name__ == "__main__":
    router = setup_router()
    vectorstore = load_vectorstore()

    test_questions = [
        "What is a group of cats called?",
        "How do I configure caching in Pydantic AI?",
        "What was the city council's vote count on the parking garage proposal?",
        "What is 2345 * 849?",
        "How many visitors were in the town of Oakhaven in September?" #expected "I don't know"
    ]

    answer_llm = create_llm(AGENT_LLM_MODEL)

    logger.info("--- Testing the Router --- \n")
    for q in test_questions:
        logger.info(f"Question: {q}")
        answer = answer_question(q, router, vectorstore, answer_llm)
        logger.info(f"Answer: {answer}\n")