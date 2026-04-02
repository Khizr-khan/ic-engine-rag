# test_retrieval.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_ic_db",
    embedding_function=embeddings
)

docs = vectorstore.similarity_search("What is compression ratio?", k=6)

print(f"Found {len(docs)} chunks\n")
for i, doc in enumerate(docs):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 0)
    print(f"Chunk {i+1} — {source} — page {page}")
    print(doc.page_content[:200])
    print("---")