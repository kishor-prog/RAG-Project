# ⚡ Production RAG Platform (Retrieval-Augmented Generation)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0+-1C3C3C.svg?style=flat&logo=chainlink&logoColor=white)](https://www.langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0467DF.svg?style=flat)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-F55036.svg?style=flat)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance, modular **Retrieval-Augmented Generation (RAG)** platform designed for document ingestion, semantic vector search, and grounded conversational question-answering for any organization, team, or codebase.

Built with **FastAPI**, **LangChain**, **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**, **FAISS Vector Database**, and **Groq LPU™ Cloud Acceleration (LLaMA-3.3-70B-Versatile)** for ultra-low latency intelligent search and synthesis.

---

## 📑 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started & Installation](#-getting-started--installation)
  - [Prerequisites](#prerequisites)
  - [Clone & Setup Virtual Environment](#1-clone--setup-virtual-environment)
  - [Configure Environment Variables](#2-configure-environment-variables)
- [Usage Guide](#-usage-guide)
  - [1. Running the FastAPI Backend Server](#1-running-the-fastapi-backend-server)
  - [2. Running the Interactive CLI Tool](#2-running-the-interactive-cli-tool)
  - [3. Running the Benchmark & Evaluation Pipeline](#3-running-the-benchmark--evaluation-pipeline)
  - [4. Running Automated Unit Tests](#4-running-automated-unit-tests)
- [API Reference](#-api-reference)
  - [System Health Check](#get-)
  - [Upload & Index PDF Document](#post-apiv1upload)
  - [Query Grounded Knowledge Base](#post-apiv1query)
- [Evaluation & Benchmarking Metrics](#-evaluation--benchmarking-metrics)
- [Security & Best Practices](#-security--best-practices)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏛 Architecture Overview

```
                      +-----------------------------+
                      |     Input PDF Document      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   PyPDFLoader & Text Chunk  |
                      | (RecursiveCharacterSplitter)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  HuggingFace Embeddings     |
                      |  (all-MiniLM-L6-v2: 384-dim)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  FAISS Local Vector Store   |
                      |    (Similarity Search k=3)  |
                      +--------------+--------------+
                                     |
    +--------------------------------+--------------------------------+
    |                                                                 |
    v                                                                 v
+------------------------+                                 +-----------------------+
|  FastAPI Async REST API|                                 | Interactive CLI Tool  |
|      (Port 8000)       |                                 |       (app.py)        |
+-----------+------------+                                 +-----------+-----------+
            |                                                          |
            +----------------------------+-----------------------------+
                                         |
                                         v
                         +-------------------------------+
                         | Groq LPU™ Cloud Acceleration  |
                         |   (LLaMA-3.3-70B-Versatile)   |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Grounded Factual AI Response  |
                         +-------------------------------+
```

---

## 🚀 Key Features

- **⚡ Blazing-Fast Inference**: Ultra-low latency responses utilizing **Groq LPUs** with `llama-3.3-70b-versatile`.
- **🔍 Dense Semantic Search**: Powered by `sentence-transformers/all-MiniLM-L6-v2` embeddings and local **FAISS** vector indexing.
- **🛡 Grounded & Hallucination-Free**: Strict contextual prompt constraints ensuring the model only answers from verified document context.
- **🌐 Universal REST API**: Async **FastAPI** application with CORS middleware, Pydantic data schemas, and auto-generated Swagger UI (`/docs`).
- **💻 Interactive CLI**: Clean terminal interface (`app.py`) for live querying and document exploration without spinning up web services.
- **📊 Evaluation & Benchmark Framework**: Built-in verification script (`eval_test.py`) computing **Context Recall**, **Faithfulness**, **Context Precision**, and **F1 Score**.
- **🧪 Automated Test Suite**: Integrated test client ensuring API integrity, payload validation, and system health.

---

## 🛠 Tech Stack

| Component | Technology / Library | Description |
| :--- | :--- | :--- |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance asynchronous REST API |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | Lightning-fast ASGI web server |
| **LLM Inference** | [Groq Cloud](https://groq.com/) | LLaMA 3.3 70B Versatile running on Groq LPUs |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Document loading, chunking, and embedding pipelines |
| **Embeddings** | [HuggingFace](https://huggingface.co/) | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database**| [FAISS](https://github.com/facebookresearch/faiss) | In-memory & persisted vector similarity search |
| **Evaluation** | [Pandas](https://pandas.pydata.org/) / Requests | Automated metrics verification and CSV report generator |

---

## 📂 Project Directory Structure

```plaintext
rag-project/
├── .env.example                     # Environment variables configuration template
├── .gitignore                       # Git ignore file (excludes secrets and virtualenv)
├── app.py                           # Standalone Interactive CLI RAG terminal application
├── main.py                          # Production FastAPI REST backend application
├── eval_test.py                     # Automated RAG benchmarking and evaluation script
├── requirements.txt                 # Project dependencies
├── advanced_evaluation_report.csv   # Latest benchmark and evaluation output report
├── model_evaluation_report.csv      # Model performance log
├── data/                            # Raw PDF storage directory
│   └── evolvex.pdf                  # Sample indexed document
├── faiss_index/                     # Persisted FAISS vector database
│   ├── index.faiss
│   └── index.pkl
└── tests/                           # Automated API test suite
    └── test_api.py
```

---

## 🏁 Getting Started & Installation

### Prerequisites

- **Python 3.10+** (Tested on Python 3.12)
- A **Groq Cloud API Key** (Free tier available at [console.groq.com](https://console.groq.com/keys))

### 1. Clone & Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/kishor-prog/RAG-Project.git
cd RAG-Project

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory by copying `.env.example`:

```bash
# Copy example configuration
cp .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

---

## 🖥 Usage Guide

### 1. Running the FastAPI Backend Server

Launch the asynchronous API server with auto-reload:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Once running:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

### 2. Running the Interactive CLI Tool

To query your indexed documents directly via the terminal:

```bash
# Query the default PDF
python app.py

# Or supply any custom PDF path
python app.py "data/your_document.pdf"
```

Example CLI session:
```text
=======================================================
   Universal RAG Interactive Terminal (Groq + LLaMA 3.3)
=======================================================
Type your question and press Enter. Type 'exit' or 'quit' to quit.

Ask a question: What is the main summary of the document?
[*] Searching vector database for relevant context...
[*] Generating answer with llama-3.3-70b-versatile...

Answer:
Based on the document context, here is the detailed summary...
--------------------------------------------------
```

---

### 3. Running the Benchmark & Evaluation Pipeline

Benchmark your RAG system's accuracy against test datasets:

```bash
python eval_test.py
```

This generates an evaluation report (`advanced_evaluation_report.csv`) with full performance metrics.

---

### 4. Running Automated Unit Tests

Run the test suite to verify endpoints and validation logic:

```bash
python -m unittest discover tests
```

---

## 📡 API Reference

### `GET /`
**Description:** Health check endpoint to verify system status and vector store availability.

#### Response (`200 OK`)
```json
{
  "status": "online",
  "engine": "Groq",
  "model": "llama-3.3-70b-versatile",
  "vector_store_initialized": true
}
```

---

### `POST /api/v1/upload`
**Description:** Upload and index a PDF document into the FAISS vector database.

#### Request (`multipart/form-data`)
- `file`: PDF file binary (`.pdf`)

#### Example cURL
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/sample.pdf"
```

#### Response (`201 Created`)
```json
{
  "message": "PDF document uploaded and indexed successfully.",
  "filename": "sample.pdf",
  "chunks_created": 3,
  "storage": "FAISS Index Saved"
}
```

---

### `POST /api/v1/query`
**Description:** Ask questions against the indexed vector database.

#### Request Body (`application/json`)
```json
{
  "question": "What are the key highlights of the document?",
  "email": "user@example.com",
  "phone_number": "+1234567890",
  "top_k": 3
}
```

#### Example cURL
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key highlights of the document?",
    "top_k": 3
  }'
```

#### Response (`200 OK`)
```json
{
  "question": "What are the key highlights of the document?",
  "answer": "The key highlights include...",
  "contexts": [
    "Context snippet 1...",
    "Context snippet 2..."
  ],
  "retrieved_count": 2,
  "model": "llama-3.3-70b-versatile"
}
```

---

## 📊 Evaluation & Benchmarking Metrics

The evaluation module (`eval_test.py`) assesses the **RAG Triad**:

| Metric | Target | Formula / Description |
| :--- | :--- | :--- |
| **Context Recall** | `> 0.85` | Measures if ground truth facts were retrieved in vector chunks: $\frac{|\text{GroundTruth} \cap \text{Context}|}{|\text{GroundTruth}|}$ |
| **Faithfulness** | `> 0.90` | Verifies that the model's answer is strictly derived from the context: $\frac{|\text{Answer} \cap \text{Context}|}{|\text{Answer}|}$ |
| **Context Precision** | `1.0` | Assesses if retrieved chunks are relevant to the query intent. |
| **Lexical F1 Score** | `> 0.80` | Harmonic mean of token precision and recall between ground truth and answer. |

---

## 🔒 Security & Best Practices

- **Secret Management**: API keys are securely loaded from environment variables (`.env`). No secrets are committed into version control.
- **CORS Protection**: Configurable middleware for safe web application communication.
- **Input Sanitization**: Pydantic schema validation blocks malformed queries and non-PDF uploads.
- **Safe Deserialization**: FAISS index persistence isolated to trusted local environments.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
