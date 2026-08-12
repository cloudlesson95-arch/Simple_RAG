import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_questions(filepath):
    """Parse the question.txt file into a list of dictionaries"""

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
    print("Loading vector database")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    questions = load_questions("baseline/question.txt")
    print(f"Loaded {len(questions)} test questions.\n")

    successful_retrievals = 0

    judge_llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0,
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )
    
    for i,q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Testing: '{q['query']}'")

        results = retriever.invoke(q["query"])
        
        context_text = "\n---\n".join([doc.page_content for doc in results])

        judge_prompt = f"""You are an impartial judge evaluating a Search engine.
        The user asked: '{q['query']}'
        The system retrieved this context:
        {context_text}

        Does the retrieved context contain the information necessary to answer the question?
        Reply ONLY with "YES" or "NO". Do not explain."""

        response = judge_llm.invoke(judge_prompt)
        decision = response.content.strip().upper()

        if "YES" in decision:
            print("\tSUCCESS: LLM says YES")
            successful_retrievals +=1
        else:
            print("\tFAIL: LLM says NO")
            print(f"\tExpected to find: {q['expected_context'][:100]}...")
            print("\t----- What was retrieved(first 100 chars): -----")
            for j, doc in enumerate(results):
                clean_chunk = repr(doc.page_content[:100].replace('\n', ' ').replace('\r', ''))
                print(f"\tChunk {j+1} (Source {doc.metadata['source']}): {clean_chunk}...")
            print("\t----------")


    precision = (successful_retrievals/len(questions)) * 100
    print(f"\nFinal score: Precision@4 - {precision:.1f}%")
    print(f"({successful_retrievals} out of {len(questions)} retrieved correctly)")

if __name__ == "__main__":
    run_evaluation()