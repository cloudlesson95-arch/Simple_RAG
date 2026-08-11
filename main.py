import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents():
    data_dir = "data"
    files = ["fictional_text.txt", "cat-facts.txt", "pydantic.llms-full.txt"]

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

if __name__ == "__main__":
    chunks = load_and_chunk_documents()

    print("\nFirst chunk:")
    print("Metadata:", chunks[0].metadata)
    print("Content:", repr(chunks[0].page_content[:100] + "..."))