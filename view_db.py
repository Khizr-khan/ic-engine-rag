from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_ic_db",
    embedding_function=embeddings
)

collection = vectorstore._collection

# include embeddings this time
data = collection.get(include=["documents", "metadatas", "embeddings"])

total = len(data['documents'])
print(f"Total chunks stored: {total}")
print("=" * 60)

for i, (doc, meta, vector) in enumerate(zip(
    data['documents'],
    data['metadatas'],
    data['embeddings']
)):
    print(f"\nChunk {i+1} of {total}")
    print("-" * 60)

    # TEXT
    print(f"File    : {meta.get('source', 'unknown')}")
    print(f"Page    : {meta.get('page', '?')}")
    print(f"Text    : {doc[:300]}...")

    # VECTOR
    print(f"Vector length : {len(vector)} numbers")
    print(f"First 10 nums : {[round(n, 4) for n in vector[:10]]}")
    print(f"Last  10 nums : {[round(n, 4) for n in vector[-10:]]}")

    print()

    if i >= 4:  # show first 5 chunks only — change this number to see more
        remaining = total - 5
        print(f"... {remaining} more chunks not shown")
        print(f"Change 'if i >= 4' to 'if i >= {total-1}' to see all")
        break