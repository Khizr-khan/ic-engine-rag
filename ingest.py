import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


INDEX_KEYWORDS = [
    "INDEX",
    "S.NO",
    "PAGE.NO",
    "Week 1",
    "TOPICS",
    "Table of Contents",
    "CONTENTS",
]

def is_noise_page(text: str, page_num: int) -> bool:
    text = text.strip()

    # Skip very short pages
    if len(text) < 100:  # reduced from 200
        return True

    # Skip pages containing index keywords
    for keyword in INDEX_KEYWORDS:
        if keyword in text:
            return True

    # Skip pages where more than 70% of lines are very short
    # Changed from 40% to 70% — Ganesan has many equation lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        short_lines = [l for l in lines if len(l) < 60]
        if len(short_lines) / len(lines) > 0.95:  # increased to 0.95 — keep formula pages
            return True

    return False

def ingest_documents(docs_dir: str = "./ic_engine_docs"):

    print(f"Loading PDFs from {docs_dir}...")
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()

    if not documents:
        print("No PDFs found! Make sure you put PDFs in ic_engine_docs/")
        return 0, 0

    print(f"Total pages loaded: {len(documents)}")

    # Filter out noise pages
    filtered = []
    skipped = 0
    for doc in documents:
        page_num = doc.metadata.get("page", 0)
        text = doc.page_content
        if is_noise_page(text, page_num):
            skipped += 1
        else:
            filtered.append(doc)

    print(f"Skipped {skipped} noise pages (index, TOC, blank)")
    print(f"Keeping {len(filtered)} clean content pages")
    documents = filtered

    if not documents:
        print("No content pages left after filtering!")
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks from {len(documents)} pages")

    print("Creating embeddings... (first time may take a few minutes)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_ic_db"
    )

    # vectorstore.persist()
    print("Done! Vector DB saved to ./chroma_ic_db")

    return len(chunks), len(set(d.metadata["source"] for d in documents))

if __name__ == "__main__":
    chunks, files = ingest_documents()
    print(f"Summary: {files} file(s) -> {chunks} chunks indexed")