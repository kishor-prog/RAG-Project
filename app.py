import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_PDF_PATH = "data/evolvex.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


def check_environment():
    """Verify essential environment variables."""
    if not GROQ_API_KEY:
        print("\n[ERROR] GROQ_API_KEY is not set.")
        print("Please create a .env file based on .env.example and set your GROQ_API_KEY.")
        sys.exit(1)


def process_pdf_to_chunks(pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 75):
    """Load a PDF document and split it into semantic chunks."""
    if not os.path.exists(pdf_path):
        print(f"\n[Error] The file '{pdf_path}' does not exist!")
        sys.exit(1)

    print(f"[*] Extracting text from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"[+] Generated {len(chunks)} text chunks from your document.")
    return chunks


def create_vectorstore(chunks):
    """Generate vector embeddings and construct a FAISS index in memory."""
    print(f"[*] Initializing embedding model ({EMBEDDING_MODEL_NAME})...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("[*] Building vector database index in FAISS...")
    vectordb = FAISS.from_documents(chunks, embedding_model)
    print("[+] Vector store built successfully.")
    return vectordb


def query_rag_system(vectordb, top_k: int = 3):
    """Interactive loop to query the RAG system using Groq and FAISS."""
    print("[*] Connecting to Groq Inference Engine...")
    client = Groq(api_key=GROQ_API_KEY.strip())

    print("\n=======================================================")
    print("   Universal RAG Interactive Terminal (Groq + LLaMA 3.3)")
    print("=======================================================")
    print("Type your question and press Enter. Type 'exit' or 'quit' to quit.\n")

    while True:
        try:
            question = input("\n\033[1;34mAsk a question\033[0m: ")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break

        if question.lower().strip() in ["exit", "quit", "q"]:
            print("Exiting system. Goodbye!")
            break

        if not question.strip():
            continue

        print("[*] Searching vector database for relevant context...")
        retrieved_docs = vectordb.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
Answer the question using ONLY the context provided below.
If the answer cannot be determined strictly from the context, reply exactly: "I don't know."

Context:
{context}

Question:
{question}
"""
        print(f"[*] Generating answer with {GROQ_MODEL}...")
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content
            print("\n\033[1;32mAnswer:\033[0m")
            print(answer)
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"\n[ERROR] Inference failed: {e}")


if __name__ == "__main__":
    check_environment()
    
    # Allow custom PDF via command-line argument if provided
    pdf_target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    
    chunks = process_pdf_to_chunks(pdf_target)
    vectordb = create_vectorstore(chunks)
    query_rag_system(vectordb)