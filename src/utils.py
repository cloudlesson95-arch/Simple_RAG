import os

def create_llm(provider: str):
    """Factory function to create LLM instances based on the provider name.
    
    Args:
        provider: The name of the LLM provider ('gemini' or 'groq').
        
    Returns:
        The appropriate LLM instance for the specified provider.
        
    Raises:
        ValueError: If an unknown provider is specified.
    """
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model = "openai/gpt-oss-20b",
            temperature=0,
            groq_api_key = os.getenv("GROQ_API_KEY")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")