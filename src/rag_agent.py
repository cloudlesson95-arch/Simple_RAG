from pydantic import BaseModel, Field
from typing import Literal
from src.config import K_RETRIEVAL, MAX_RETRIES, EMBEDDING_LOCAL_MODEL, MAIN_LLM_MODEL, CHROMA_PERSIST_DIR
from src.utils import create_llm

from src.logging_config import setup_logging
logger = setup_logging(__name__)

class RouteDecision(BaseModel):
    """Decides which data source to use based on the user's query."""

    source: Literal["pydantic.llms-full.txt", "cat-facts.txt", "fictional_text.txt", "none"] = Field(
        description="""Which data source:
        - 'pydantic.llms-full.txt': questions about Python, AI, Pydantic, Anthropic, LLMs, agents, models and their values and parameters 
        - 'cat-facts.txt': questions about cats, animals, pets
        - 'fictional_text.txt': questions about Oakhaven, pumpkin festival, city council, Elena Rostova
        - 'none': basic knowledge, regular math, greetings, or anything not in the above"""
    )

    reasoning: str = Field(
        description="A short 1-sentence explanation of why you chose this source."
    )

def setup_router():
    """Setup the router LLM for determining data sources.
    
    Returns:
        The router LLM configured with structured output for RouteDecision.
    """    
    llm = create_llm(MAIN_LLM_MODEL)

    router_llm = llm.with_structured_output(RouteDecision)
    return router_llm

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
        logger.debug(f"Current context_text: {context_text}")

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
            logger.debug(f"Current answer: {answer}")

            rephrase_prompt = f"""The user asked: "{current_query}"
            A search returned no useful results. 
            Rephrase this question using different keywords.
            Return ONLY the rephrased question."""

            current_query = answer_llm.invoke(rephrase_prompt).content.strip()
            logger.info(f"[Agent] Rephrased query: '{current_query}'")

            answer = retrieve_and_answer(current_query)

    return answer
