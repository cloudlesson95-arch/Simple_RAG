import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
#for local embedding
from langchain_huggingface import HuggingFaceEmbeddings #pip install sentence-transformers langchain-huggingface #~80MB model + torch

load_dotenv()

def load_and_chunk_documents():
    data_dir = "data"
    # files = ["fictional_text.txt", "cat-facts.txt"]
    files = ["fictional_text.txt", "cat-facts.txt", "pydantic.llms-full.txt"] #pydantic can use a lot of limits. Use only for final checks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    all_chunks = []

    for filename in files:
        filepath = os.path.join(data_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.create_documents([text], metadatas=[{"source": filename}])    
        all_chunks.extend(chunks)

        print(f"Loaded {filename}: Split into {len(chunks)}")

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def create_or_load_vectorstore(chunks):
    """Embeds the chunks and stores them in ChromaDB"""

    # #gemini model through API
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model = "models/gemini-embedding-001",
    #     google_api_key=os.getenv("GOOGLE_API_KEY")
    # )

    #local embedding model
    #make sure to remove 'chroma_db' folder on switch to avoid different embedding error 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        print(f"Loading existing database from {persist_directory}...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"Creating new database at {persist_directory}. This might take a while...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )

    return vectorstore


if __name__ == "__main__":
    chunks = load_and_chunk_documents()

    vectorstore = create_or_load_vectorstore(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    test_question = "What is a group of cats called?"
    print(f"\nSearching for: {test_question}")
    results = retriever.invoke(test_question)

    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    context_text = "\n---\n".join([doc.page_content for doc in results])

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
    If the anwer is not in the context, say "I don't know based on the context."
    
    Context:
    {context_text}

    Question: {test_question}

    Answer:"""

    print("\nSending to LLM")

    response = llm.invoke(prompt)
    print(response.content)