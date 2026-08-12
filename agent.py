import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq #back-up llm

load_dotenv()

class RouteDecision(BaseModel):
    """Decides which data source to use based on the user's query."""

    source: Literal["pydantic.llms.txt", "cat-facts.txt", "fictional_text.txt", "none"] = Field(
        description="""Which data source:
        - 'pydantic.llms.txt': questions about Python, AI, Pydantic, LLMs, agents, models
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

if __name__ == "__main__":
    router = setup_router()

    test_questions = [
        "What is a group of cats called?",
        "How do I configure caching in Pydantic AI?",
        "What was the city council's vote count on the parking garage proposal?",
        "What is 2345 * 849?"
    ]

    print("--- Testing the Router --- \n")
    for q in test_questions:
        print(f"Question: {q}")
        decision = router.invoke(q)
        print(f" Routed to: {decision.source}")
        print(f" Reasoning: {decision.reasoning}\n")