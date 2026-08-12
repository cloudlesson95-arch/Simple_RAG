import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq #back-up llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

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
    # llm = ChatGoogleGenerativeAI(
    #     model = "gemini-2.5-flash",
    #     temperature = 0,
    #     google_api_key = os.getenv("GOOGLE_API_KEY")
    # )
    llm = ChatGroq(
	model = "llama-3.1-8b-instant",
	temperature = 0,
	groq_api_key = os.getenv("GROK_API_KEY")
    )

    router_llm = llm.with_structured_output(RouteDecision)
    return router_llm

def load_vectorstore():
    print("Loading vector database")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore


def answer_question(question, router, vectorstore, answer_llm):
    decision = router.invoke(question)
    print(f"[Router]: '{decision.source}' ({decision.reasoning})")

    if decision.source == "none":
        print("[Router] No retrieval needed")
        response = answer_llm.invoke(question)
        return response.content

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"source": decision.source}
        }
    )
    results = retriever.invoke(question)
    context_text = "\n---\n".join([doc.page_content for doc in results])

    prompt = f"""Answer ONLY based on the provided context.
    If the answer is not in the context, say "I don't know."

    Context:
    {context_text}

    Question: {question}
    Answer:"""

    response = answer_llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    router = setup_router()
    vectorstore = load_vectorstore()

    test_questions = [
        "What is a group of cats called?",
        "How do I configure caching in Pydantic AI?",
        "What was the city council's vote count on the parking garage proposal?",
        "What is 2345 * 849?"
    ]

    answer_llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature = 0,
        groq_api_key = os.getenv("GROK_API_KEY")
    )

    print("--- Testing the Router --- \n")
    for q in test_questions:
        print(f"Question: {q}")
        answer = answer_question(q, router, vectorstore, answer_llm)
        print(f"Answer: {answer}\n")