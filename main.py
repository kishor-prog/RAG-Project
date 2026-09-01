import os
import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

# =====================================================
# Configuration & Setup
# =====================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("rag-platform")

DATA_DIR = os.getenv("DATA_DIR", "data")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

EMBEDDING_MODEL = None

# =====================================================
# FastAPI Application Initialization
# =====================================================

app = FastAPI(
    title="Production RAG API Platform",
    description=(
        "High-performance Retrieval-Augmented Generation (RAG) backend engine for document intelligence and Q&A, "
        "powered by FAISS Vector Database, LangChain, HuggingFace embeddings, and ultra-fast Groq LLaMA-3.3 inference."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for all origins in development / production web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Embedding Model (Lazy Loaded Singleton)
# =====================================================

def get_embedding_model():
    """Retrieve or lazily initialize the HuggingFace embedding model."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return EMBEDDING_MODEL


# =====================================================
# Pydantic Schemas
# =====================================================

class HealthResponse(BaseModel):
    status: str = Field(..., example="online")
    engine: str = Field(..., example="Groq")
    model: str = Field(..., example="llama-3.3-70b-versatile")
    vector_store_initialized: bool = Field(..., example=True)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, example="What are the key highlights of the document?")
    email: Optional[str] = Field(None, example="user@example.com")
    phone_number: Optional[str] = Field(None, example="+1234567890")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    retrieved_count: int
    model: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    storage: str


# =====================================================
# API Endpoints
# =====================================================

@app.get("/", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns the operational status of the RAG engine and vector store availability."""
    index_exists = os.path.exists(FAISS_INDEX_PATH)
    return HealthResponse(
        status="online",
        engine="Groq",
        model=DEFAULT_MODEL,
        vector_store_initialized=index_exists
    )


@app.post(
    "/api/v1/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Document"]
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and index a PDF document:
    1. Saves the PDF file to the data storage directory.
    2. Parses and splits the text into semantic chunks using RecursiveCharacterTextSplitter.
    3. Builds and persists the FAISS vector index locally.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF documents (.pdf) are supported."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    # Save PDF locally
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        logger.info(f"Saved uploaded PDF: {file_path}")
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save PDF on server: {str(e)}"
        )

    # Load and split PDF
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=75
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} text chunks from {file.filename}")
    except Exception as e:
        logger.error(f"PDF Parsing Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF document: {str(e)}"
        )

    # Build and persist FAISS index
    try:
        embeddings = get_embedding_model()
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(FAISS_INDEX_PATH)
        logger.info(f"FAISS vector store persisted to {FAISS_INDEX_PATH}")
    except Exception as e:
        logger.error(f"Vector Store Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate vector embeddings or store FAISS index: {str(e)}"
        )

    return UploadResponse(
        message="PDF document uploaded and indexed successfully.",
        filename=file.filename,
        chunks_created=len(chunks),
        storage="FAISS Index Saved"
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Conversational AI"]
)
async def query_rag(request: QueryRequest):
    """
    Query the knowledge base using Retrieval-Augmented Generation:
    1. Queries the local FAISS vector store using semantic search.
    2. Constructs a grounded context prompt.
    3. Requests grounded synthesis via Groq's high-speed LLaMA-3.3-70B model.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The question string cannot be blank."
        )

    if not os.path.exists(FAISS_INDEX_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector database index not found. Please upload a PDF to index documents first."
        )

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable is missing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY not configured on server."
        )

    try:
        # Load embedding model & FAISS index
        embeddings = get_embedding_model()
        vector_db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Retrieve similar chunks
        top_k = request.top_k or 3
        retrieved_docs = vector_db.similarity_search(cleaned_question, k=top_k)
        contexts = [doc.page_content for doc in retrieved_docs]
        context_str = "\n\n".join(contexts)

        # Grounded prompt instruction
        prompt = f"""
Answer the question using ONLY the context below.
If the answer is not available in the context, reply exactly: "I don't know."

Context:
{context_str}

Question:
{cleaned_question}
"""

        # Query Groq API
        client = Groq(api_key=GROQ_API_KEY.strip())
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content

        logger.info(
            f"[QUERY PROCESSED] Question: '{cleaned_question}' | "
            f"Email: {request.email} | "
            f"Phone: {request.phone_number} | "
            f"Retrieved: {len(contexts)} chunks"
        )

        return QueryResponse(
            question=cleaned_question,
            answer=answer,
            contexts=contexts,
            retrieved_count=len(contexts),
            model=DEFAULT_MODEL
        )

    except Exception as e:
        logger.error(f"RAG Inference Exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG Processing Error: {str(e)}"
        )