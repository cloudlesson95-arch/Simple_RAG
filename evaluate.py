import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    questions = load_questions("baseline/question.txt")
    print(f"Loaded {len(questions)} test questions.\n")

    successful_retrievals = 0

    for i,q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Testing: '{q['query']}'")

        results = retriever.invoke(q["query"])

        found_in_top_k = False
        for doc in results:
            #A simple substring match if the expected context is inside the chunk content. Won't always work perfectly
            if q["expected_context"].lower() in doc.page_content.lower():
                found_in_top_k = True
                break

        if found_in_top_k:
            print("\tSUCCESS: Expected context found in Top-4")
            successful_retrievals +=1
        else:
            print("\tFAIL: Expected context NOT found in Top-4")
            print(f"\tExpected to find: {q['expected_context'][:100]}...")

        precision = (successful_retrievals/len(questions)) * 100
        print(f"\nFinal score: Precision@4 - {precision:.1f}%")
        print(f"({successful_retrievals} out of {len(questions)} retrieved correctly)")

if __name__ == "__main__":
    run_evaluation()